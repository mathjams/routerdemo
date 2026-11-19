"""Train the router and its branches on CIFAR-100.

    python -m training.train        (from the repo root; a GPU is strongly recommended)

Every outer step has two phases:
  1. Branch phase. Each batch is split between the branches by sampling from the router
     (plus a little epsilon exploration), and a branch takes a gradient step whenever it has
     collected a full batch. Each sample's loss is weighted by a "gate" that shrinks when the
     router's routing of that sample disagrees with what the branch losses say is best.
  2. Router phase. With the branches frozen, the router learns to imitate a teacher that
     prefers the branch with the lowest loss + cost on each sample, plus a load-balancing
     term that keeps every branch in use.

Everything is written to runs/<timestamp>/: log.txt, checkpoints/, and weights/ in the
format the demo loads (copy weights/*.pth into backend/weights/ to serve them).
"""
import os
import random
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn

from backend.models import RouterModel, SimpleCNN

from .data import CUDAPrefetcher, TensorQueue, get_cifar100_dataloaders, infinite_loader
from .evaluate import evaluate
from .router_system import RouterSystem

# One row per branch: (base_channels, num_layers, cost). The router pays `cost` for choosing it.
BRANCHES = [
    (64, 4, 0.00),
    (88, 4, 0.10),
    (64, 5, 0.25),
    (80, 5, 0.45),
    (96, 5, 0.70),
]


def seed_all(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_module(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()


def save_checkpoint(network, branch_optimizers, router_optimizer, step, path, hyperparams):
    torch.save({
        "step": step,
        "router_state": network.router.state_dict(),
        "branch_states": [b.state_dict() for b in network.branches],
        "router_opt_state": router_optimizer.state_dict(),
        "branch_opt_states": [opt.state_dict() for opt in branch_optimizers],
        "costs": network.costs.detach().cpu(),
        "hyperparams": hyperparams,
    }, path)


def main(total_steps=4000, step_branches=100, step_router=300, batch_size=128, num_workers=2,
         checkpoint_every=10, validate_every=50):
    """Run training. `step_branches` / `step_router` are the batches per phase in each outer step.

    A checkpoint holds every model and optimizer (~450 MB with the default branches), and one
    is kept per `checkpoint_every` steps, so a full run needs a lot of disk.
    """
    base_dir = os.environ.get("BASE_DIR", os.getcwd())
    run_dir = os.path.join(base_dir, "runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    def log(message):
        print(message)
        with open(os.path.join(run_dir, "log.txt"), "a") as f:
            f.write(message + "\n")

    seed_all(0)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- Hyperparameters ----
    lr = 0.0001  # branches
    lr_router = 0.0002
    lr_epsilon = 0.1  # a branch's LR is lr / (its share of the data + lr_epsilon); stops it exploding
    gamma = 0.05  # EMA rate for each branch's share of the data
    eps = 0.03  # routing exploration; decays every outer step
    cf = 1.0  # gate = exp(-cf * router loss), clamped below at gate_min
    gate_min = 0.1
    beta = 20  # sharpness of the teacher distribution; decays every outer step
    ls = 0.0  # label smoothing on the teacher
    t_router = 1.0  # router softmax temperature
    lb_weight = 1.0  # load-balancing weight; decays every outer step
    router_reset_steps = 3000  # for this many outer steps, the router is re-initialised before its phase

    N = len(BRANCHES)
    log("=" * 60)
    log(f"Router training - device: {device}, GPUs: {num_gpus}, branches: {N}, steps: {total_steps}")
    log(f"Run directory: {run_dir}")
    log("=" * 60)

    train_loader, val_loader = get_cifar100_dataloaders(
        os.path.join(base_dir, "data"), batch_size=batch_size, num_workers=num_workers
    )
    log(f"Train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # With several GPUs the branches are spread across them; the router stays on the first.
    branch_devices = [torch.device(f"cuda:{i % num_gpus}") if num_gpus > 1 else device for i in range(N)]
    branches = [SimpleCNN(100, ch, layers).to(branch_devices[i]) for i, (ch, layers, _) in enumerate(BRANCHES)]
    for i, b in enumerate(branches):
        log(f"Branch {i}: {sum(p.numel() for p in b.parameters()) / 1e6:.2f}M params on {branch_devices[i]}")

    router_model = RouterModel(N).to(device)
    log(f"Router: {sum(p.numel() for p in router_model.parameters()) / 1e6:.2f}M params")

    costs = torch.tensor([cost for _, _, cost in BRANCHES], device=device)
    network = RouterSystem(router_model, branches, costs)

    branch_optimizers = [torch.optim.AdamW(b.parameters(), lr=lr) for b in branches]
    router_optimizer = torch.optim.AdamW(router_model.parameters(), lr=lr_router)

    # Samples routed to a branch wait here until there are enough for a full batch.
    bufs_x = [TensorQueue(deque()) for _ in range(N)]
    bufs_y = [TensorQueue(deque()) for _ in range(N)]
    bufs_gate = [TensorQueue(deque()) for _ in range(N)]

    x_ema = [1.0 / N] * N  # running share of the data each branch receives

    # Running statistics for the log, reset at every checkpoint.
    total_routed, total_seen = [0] * N, 0
    r_loss, r_acc, r_steps = 0.0, 0.0, 0
    b_loss, b_acc, b_count, b_lr = [0.0] * N, [0.0] * N, [0] * N, [0.0] * N

    if device.type == "cuda":
        get_batch = CUDAPrefetcher(train_loader, device).next
    else:
        train_iter = infinite_loader(train_loader)

        def get_batch():
            x, y = next(train_iter)
            return x.to(device), y.to(device)

    for step in range(total_steps):
        eps *= 0.9995
        beta *= 0.999
        lb_weight *= 0.9995

        # ---- Branch phase ----
        for _ in range(step_branches):
            x, y = get_batch()
            router_loss_ps = network.router_instaloss(x, y, cost_weight=1.0, t_router=t_router, beta=beta)  # (B,)
            gate = torch.exp(-cf * router_loss_ps).clamp(min=gate_min)  # (B,)
            routes = network.route(x, temperature=1.0, epsilon=eps)  # (B,)

            frac = (torch.bincount(routes, minlength=N).float() / float(x.size(0))).tolist()
            gamma_batch = 1.0 - (1.0 - gamma) ** float(x.size(0))
            for i in range(N):
                x_ema[i] = (1.0 - gamma_batch) * x_ema[i] + gamma_batch * float(frac[i])

            for i in range(N):
                mask = routes == i
                cnt = int(mask.sum().item())
                total_routed[i] += cnt
                if cnt > 0:
                    bufs_x[i].add(x[mask])
                    bufs_y[i].add(y[mask])
                    bufs_gate[i].add(gate[mask])
            total_seen += int(x.size(0))

            for i in range(N):
                while bufs_x[i].total >= batch_size:
                    batch_x = bufs_x[i].pop(batch_size).to(branch_devices[i], non_blocking=True)
                    batch_y = bufs_y[i].pop(batch_size).to(branch_devices[i], non_blocking=True)
                    batch_gate = bufs_gate[i].pop(batch_size).to(branch_devices[i], non_blocking=True)

                    lr_eff = lr / (x_ema[i] + lr_epsilon)
                    for pg in branch_optimizers[i].param_groups:
                        pg["lr"] = lr_eff

                    branches[i].train()
                    branch_optimizers[i].zero_grad(set_to_none=True)
                    outputs = branches[i](batch_x)
                    per_sample_loss = nn.functional.cross_entropy(outputs, batch_y, reduction="none")  # (B,)
                    loss = (batch_gate * per_sample_loss).mean()
                    loss.backward()
                    branch_optimizers[i].step()

                    b_loss[i] += float(loss.item())
                    b_acc[i] += float((outputs.argmax(dim=1) == batch_y).float().mean().item())
                    b_count[i] += 1
                    b_lr[i] += float(lr_eff)

        # ---- Router phase (branches frozen) ----
        for b in branches:
            b.eval()
            for p in b.parameters():
                p.requires_grad = False
        if step < router_reset_steps:
            network.router.apply(reset_module)
        router_optimizer = torch.optim.AdamW(network.router.parameters(), lr=lr_router)

        for stepcount in range(step_router):
            x, y = get_batch()
            cw = 0.25 + 0.75 * (stepcount / step_router)  # the cost penalty ramps up within the phase
            stats = network.train_step(
                x, y, router_optimizer, cw, t_router=t_router, beta=beta, ls=ls, lb_weight=lb_weight
            )
            r_loss += float(stats["router_loss"])
            r_acc += float(stats["accuracy"])
            r_steps += 1

        for b in branches:
            for p in b.parameters():
                p.requires_grad = True

        hyperparams = {"eps": eps, "beta": beta, "lb_weight": lb_weight, "x_ema": x_ema}

        if (step + 1) % checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, f"ckpt_step_{step + 1:04d}.pt")
            save_checkpoint(network, branch_optimizers, router_optimizer, step + 1, path, hyperparams)
            log(f"Step {step + 1}: router loss={r_loss / max(1, r_steps):.4f}, acc={r_acc / max(1, r_steps):.3f}")
            for i in range(N):
                routed = (total_routed[i] / total_seen) if total_seen > 0 else 0.0
                if b_count[i] > 0:
                    log(
                        f"  B{i}: loss={b_loss[i] / b_count[i]:.4f}, acc={b_acc[i] / b_count[i]:.3f}, "
                        f"batches={b_count[i]}, lr={b_lr[i] / b_count[i]:.6f}, routed={routed:.2%}"
                    )
                else:
                    log(f"  B{i}: no training, routed={routed:.2%}")

            total_routed, total_seen = [0] * N, 0
            r_loss, r_acc, r_steps = 0.0, 0.0, 0
            b_loss, b_acc, b_count, b_lr = [0.0] * N, [0.0] * N, [0] * N, [0.0] * N

        if (step + 1) % validate_every == 0:
            log(f"\nValidation @ step {step + 1}")
            evaluate(router_model, branches, val_loader, log)

    log("\nFinal validation")
    evaluate(router_model, branches, val_loader, log)

    hyperparams = {"eps": eps, "beta": beta, "lb_weight": lb_weight, "x_ema": x_ema}
    save_checkpoint(
        network, branch_optimizers, router_optimizer, total_steps, os.path.join(checkpoint_dir, "final.pt"), hyperparams
    )
    torch.save(router_model.state_dict(), os.path.join(weights_dir, "router.pth"))
    for i, b in enumerate(branches):
        torch.save(b.state_dict(), os.path.join(weights_dir, f"branch_{i}.pth"))
    log(f"\nSaved weights to {weights_dir}")


if __name__ == "__main__":
    main()
