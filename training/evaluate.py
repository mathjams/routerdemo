"""Measure routed accuracy on the CIFAR-100 test set (downloaded to ./data on first use).

    python -m training.evaluate                                      # the weights the demo serves
    python -m training.evaluate --weights runs/<timestamp>/weights
    python -m training.evaluate --checkpoint runs/<timestamp>/checkpoints/final.pt
"""
import argparse

import torch

from backend.models import RouterModel, SimpleCNN, load_models

from .data import get_cifar100_dataloaders


@torch.no_grad()
def evaluate(router, branches, loader, log=print):
    """Report routed accuracy, how images spread over the branches, and each branch on its own.

    Returns the routed accuracy. Models are left in the train/eval mode they came in.
    """
    models = [router, *branches]
    modes = [model.training for model in models]
    for model in models:
        model.eval()

    device = next(router.parameters()).device
    n = len(branches)
    total = 0
    routed = torch.zeros(n)  # images sent to each branch
    routed_correct = torch.zeros(n)  # ... that it then classified correctly
    alone_correct = torch.zeros(n)  # correct answers if this branch handled every image
    oracle_correct = 0  # images that at least one branch gets right

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        routes = router(x).argmax(dim=-1)
        correct = torch.stack([
            branch(x.to(next(branch.parameters()).device)).argmax(dim=-1).to(device) == y
            for branch in branches
        ]).float()  # (branches, B)
        chosen = correct.gather(0, routes.unsqueeze(0)).squeeze(0)  # was the routed branch right?

        total += y.numel()
        routed += torch.bincount(routes, minlength=n).cpu()
        routed_correct += torch.bincount(routes, weights=chosen, minlength=n).cpu()
        alone_correct += correct.sum(dim=1).cpu()
        oracle_correct += correct.amax(dim=0).sum().item()

    for model, mode in zip(models, modes):
        model.train(mode)

    accuracy = routed_correct.sum().item() / total
    log(f"Routed accuracy: {accuracy:.4f}   (with perfect routing: {oracle_correct / total:.4f})")
    for i in range(n):
        on_routed = f"{routed_correct[i] / routed[i]:.4f}" if routed[i] > 0 else "n/a"
        log(
            f"  Branch {i}: gets {routed[i] / total:6.2%} of images, accuracy {on_routed} on those, "
            f"{alone_correct[i] / total:.4f} if it handled every image"
        )
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate routed inference on the CIFAR-100 test set.")
    parser.add_argument("--weights", default="backend/weights", help="directory with router.pth and branch_<i>.pth")
    parser.add_argument("--checkpoint", help="evaluate a training checkpoint instead of a weights directory")
    parser.add_argument("--data", default="data", help="where CIFAR-100 lives (downloaded if missing)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        router = RouterModel.from_state_dict(checkpoint["router_state"])
        branches = [SimpleCNN.from_state_dict(state) for state in checkpoint["branch_states"]]
    else:
        router, branches = load_models(args.weights)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader = get_cifar100_dataloaders(args.data, batch_size=args.batch_size)
    evaluate(router.to(device), [branch.to(device) for branch in branches], test_loader)


if __name__ == "__main__":
    main()
