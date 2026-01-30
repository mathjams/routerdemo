import os
import torch
import torch.nn as nn
from collections import deque
from router_models import SimpleCNN, RouterModel
from router_system import RouterSystem
from utils import seed_all, get_imagenet_dataloaders, infinite_loader, CUDAPrefetcher, TensorQueue, reset_module, save_checkpoint
from validation import run_validation


def main():
    # ---- Config (Colab-friendly defaults; raise if you want) ----
    BASE_DIR = os.environ.get("BASE_DIR", os.getcwd())
    IMAGENET_DIR = f"{BASE_DIR}/data"

    # Create checkpoint directory
    save_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    USE_SUBSET = False
    SUBSET_SIZE = 10000
    DOWNLOAD = True

    seed_all(0)

    # Multi-GPU setup
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = num_gpus > 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    batch_size = 128
    num_workers = 2  # Colab tends to be happier with 0-2
    lr = 0.0001
    lr_router = 0.0002

    eps = 0.03
    gamma = 0.05
    lr_epsilon = 0.1  # increased to prevent LR explosion when routing is unbalanced
    cf = 1.0  # coefficient for gate calculation
    gate_min = 0.1  # minimum gate value

    total_steps = 4000
    step_branches = 100
    step_router = 300

    beta = 20
    ls = 0.0
    t_router = 1.0
    lb_weight = 1.0

    branch_params = [
        (64, 4, 0.00),
        (88, 4, 0.10),
        (64, 5, 0.25),
        (80, 5, 0.45),
        (96, 5, 0.70),
    ]

    os.makedirs(f"{BASE_DIR}/results", exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{BASE_DIR}/results/training_{timestamp}.txt"

    def log_and_print(msg: str):
        print(msg)
        with open(results_file, "a") as f:
            f.write(msg + "\n")

    log_and_print("=" * 60)
    log_and_print(f"Router Training - Device: {device}, GPUs: {num_gpus}, Branches: {len(branch_params)}, Steps: {total_steps}")
    log_and_print(f"Multi-GPU: {use_multi_gpu}")
    log_and_print(f"Timestamp: {timestamp}")
    log_and_print("=" * 60)

    train_loader, val_loader = get_imagenet_dataloaders(
        IMAGENET_DIR,
        batch_size=batch_size,
        num_workers=num_workers,
        subset_size=(SUBSET_SIZE if USE_SUBSET else None),
        download=DOWNLOAD,
    )
    log_and_print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    N = len(branch_params)

    # Distribute branches across GPUs if available
    if use_multi_gpu:
        branch_devices = [torch.device(f"cuda:{i % num_gpus}") for i in range(N)]
        branches = [SimpleCNN(100, ch, layers).to(branch_devices[i]) for i, (ch, layers, _) in enumerate(branch_params)]
        log_and_print(f"Using {num_gpus} GPUs - distributing {N} branches across GPUs")
        for i, b in enumerate(branches):
            log_and_print(f"Branch {i}: {sum(p.numel() for p in b.parameters())/1e6:.2f}M params (GPU {i % num_gpus})")
    else:
        branch_devices = [device] * N
        branches = [SimpleCNN(100, ch, layers).to(device) for ch, layers, _ in branch_params]
        for i, b in enumerate(branches):
            log_and_print(f"Branch {i}: {sum(p.numel() for p in b.parameters())/1e6:.2f}M params")

    router_model = RouterModel(N).to(device)
    log_and_print(f"Router: {sum(p.numel() for p in router_model.parameters())/1e6:.2f}M params")

    costs = torch.tensor([cost for _, _, cost in branch_params], device=device)
    network = RouterSystem(router_model, branches, costs)

    branch_optimizers = [torch.optim.AdamW(b.parameters(), lr=lr) for b in branches]
    router_optimizer = torch.optim.AdamW(router_model.parameters(), lr=lr_router)

    bufs_x = [TensorQueue(deque()) for _ in range(N)]
    bufs_y = [TensorQueue(deque()) for _ in range(N)]
    bufs_gate = [TensorQueue(deque()) for _ in range(N)]

    total_routed, total_seen = [0] * N, 0
    x_ema = [1.0 / N] * N

    r_loss, r_acc, r_steps = 0.0, 0.0, 0
    b_loss, b_acc, b_count, b_lr = [0.0] * N, [0.0] * N, [0] * N, [0.0] * N

    if device.type == "cuda":
        prefetcher = CUDAPrefetcher(train_loader, device)
        def get_batch():
            return prefetcher.next()
    else:
        train_iter = infinite_loader(train_loader)
        def get_batch():
            x, y = next(train_iter)
            return x.to(device), y.to(device)

    # CIFAR-100 has 100 classes - using a subset for visualization
    class_names = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

    # ---- Training ----
    for step in range(total_steps):
        # Branch phase
        eps *= 0.9995
        beta *= 0.999
        lb_weight*=0.9995
        cw_branch = 1.0
        for branch_iter in range(step_branches):
            x, y = get_batch()
            with torch.no_grad():
                router_loss_ps, _, _ = network.router_instaloss(
                    x, y,
                    cost_weight=cw_branch,
                    t_router=t_router,
                    beta=beta
                )  # (B,)
                gate = torch.exp(-cf * router_loss_ps).clamp(min=gate_min)  # (B,)

                routes = network.route(x, temperature=1.0, epsilon=eps)  # (B,)

                # DEBUG: Print routing info for first iteration
                if step == 0 and branch_iter == 0:
                    print(f"[DEBUG] router_loss_ps: mean={router_loss_ps.mean().item():.3f}, min={router_loss_ps.min().item():.3f}, max={router_loss_ps.max().item():.3f}")
                    print(f"[DEBUG] gate: mean={gate.mean().item():.3f}, min={gate.min().item():.3f}, max={gate.max().item():.3f}")
                    route_counts = torch.bincount(routes, minlength=N)
                    print(f"[DEBUG] route distribution: {route_counts.tolist()}")

            with torch.no_grad():
                counts = torch.bincount(routes, minlength=N).float()
                frac = (counts / float(x.size(0))).tolist()

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
                    batch_x = bufs_x[i].pop(batch_size)
                    batch_y = bufs_y[i].pop(batch_size)
                    batch_gate = bufs_gate[i].pop(batch_size)

                    # Move data to the correct GPU for this branch
                    if use_multi_gpu:
                        batch_x = batch_x.to(branch_devices[i], non_blocking=True)
                        batch_y = batch_y.to(branch_devices[i], non_blocking=True)
                        batch_gate = batch_gate.to(branch_devices[i], non_blocking=True)

                    lr_eff = lr / (x_ema[i] + lr_epsilon)
                    for pg in branch_optimizers[i].param_groups:
                        pg["lr"] = lr_eff

                    branches[i].train()
                    branch_optimizers[i].zero_grad(set_to_none=True)

                    outputs = branches[i](batch_x)
                    per_sample_loss = nn.functional.cross_entropy(outputs, batch_y, reduction="none")  # (B,)
                    loss = (batch_gate * per_sample_loss).mean()  # <-- blame-weighted branch update

                    # DEBUG: Print training info for first few steps
                    if step == 0 and b_count[i] < 2:
                        print(f"[DEBUG] Branch {i}: loss={loss.item():.4f}, gate_mean={batch_gate.mean().item():.3f}, gate_min={batch_gate.min().item():.3f}, lr={lr_eff:.6f}")

                    loss.backward()
                    branch_optimizers[i].step()

                    with torch.no_grad():
                        acc = (outputs.argmax(dim=1) == batch_y).float().mean()

                    b_loss[i] += float(loss.item())
                    b_acc[i] += float(acc.item())
                    b_count[i] += 1
                    b_lr[i] += float(lr_eff)

        # Router phase
        for b in branches:
            b.eval()
            for p_ in b.parameters():
                p_.requires_grad = False
        if step<3000:
            network.router.apply(reset_module)
        router_optimizer = torch.optim.AdamW(network.router.parameters(), lr=lr_router)

        for stepcount in range(step_router):
            x, y = get_batch()
            cw = 0.25 + 0.75 * (stepcount / step_router)
            stats = network.train_step(
                x, y,
                router_optimizer,
                cw,
                t_router=t_router,
                beta=beta,
                ls=ls,
                lb_weight=lb_weight,
            )
            r_loss += float(stats["router_loss"])
            r_acc += float(stats["accuracy"])
            r_steps += 1

        for b in branches:
            for p_ in b.parameters():
                p_.requires_grad = True

        # checkpoint/log every 10
        if (step + 1) % 10 == 0:
            path = os.path.join(save_dir, f"ckpt_step_{step+1:04d}.pt")
            hyperparams = {"eps": eps, "beta": beta, "lb_weight": lb_weight, "x_ema": x_ema}
            save_checkpoint(network, branch_optimizers, router_optimizer, step+1, path, hyperparams)
            log_and_print(f"Step {step+1}: Router loss={r_loss/max(1,r_steps):.4f}, acc={r_acc/max(1,r_steps):.3f}")
            for i in range(N):
                frac_i = (total_routed[i] / total_seen) if total_seen > 0 else 0.0
                if b_count[i] > 0:
                    log_and_print(
                        f"  B{i}: loss={b_loss[i]/b_count[i]:.4f}, acc={b_acc[i]/b_count[i]:.3f}, "
                        f"batches={b_count[i]}, lr={b_lr[i]/b_count[i]:.6f}, routed={frac_i:.2%}"
                    )
                else:
                    log_and_print(f"  B{i}: no training, routed={frac_i:.2%}")

            r_loss, r_acc, r_steps = 0.0, 0.0, 0
            b_loss, b_acc, b_count, b_lr = [0.0] * N, [0.0] * N, [0] * N, [0.0] * N
            total_routed, total_seen = [0] * N, 0

        # ✅ validation every 50 "epochs" (outer steps)
        if (step + 1) % 50 == 0:
            log_and_print("\n" + "=" * 60)
            log_and_print(f"VAL @ step {step+1}")
            log_and_print("=" * 60)
            run_validation(
                network=network,
                branches=branches,
                val_loader=val_loader,
                device=device,
                log_and_print=log_and_print,
                N=N,
                class_names=class_names,
                BASE_DIR=BASE_DIR,
                timestamp=timestamp,
                do_visualize=False,   # keep periodic val fast
                do_class_stats=False, # keep periodic val fast
                branch_devices=branch_devices,
            )

    # ============================================
    # Final Validation + Visualization + Save
    # ============================================
    run_validation(
        network=network,
        branches=branches,
        val_loader=val_loader,
        device=device,
        log_and_print=log_and_print,
        N=N,
        class_names=class_names,
        BASE_DIR=BASE_DIR,
        timestamp=timestamp,
        do_visualize=True,
        do_class_stats=True,
        branch_devices=branch_devices,
    )

    # ---- Save models ----
    log_and_print("\n" + "=" * 60)
    log_and_print("Saving final models...")
    log_and_print("=" * 60)

    # Save final checkpoint (includes optimizer states for resuming)
    final_ckpt_path = os.path.join(save_dir, f"final_checkpoint_{timestamp}.pt")
    hyperparams = {"eps": eps, "beta": beta, "lb_weight": lb_weight, "x_ema": x_ema}
    save_checkpoint(network, branch_optimizers, router_optimizer, total_steps, final_ckpt_path, hyperparams)
    log_and_print(f"✓ Final checkpoint: {final_ckpt_path}")

    # Save individual model weights (for inference only)
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    for i in range(N):
        path = os.path.join(models_dir, f"branch_{i}_{timestamp}.pth")
        torch.save(branches[i].state_dict(), path)
        log_and_print(f"✓ Branch {i}: {path}")

    router_path = os.path.join(models_dir, f"router_{timestamp}.pth")
    torch.save(router_model.state_dict(), router_path)
    log_and_print(f"✓ Router: {router_path}")

    # Save model architecture info
    model_info = {
        "timestamp": timestamp,
        "num_branches": N,
        "branch_params": branch_params,
        "num_classes": 100,
        "total_steps": total_steps,
        "final_val_acc": None,  # Will be set from validation
    }
    info_path = os.path.join(models_dir, f"model_info_{timestamp}.pt")
    torch.save(model_info, info_path)
    log_and_print(f"✓ Model info: {info_path}")

    log_and_print(f"\n✓ All models saved to {models_dir}/")
    log_and_print(f"✓ Results saved to {results_file}")
    log_and_print("=" * 60)


if __name__ == "__main__":
    main()