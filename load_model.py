#!/usr/bin/env python3
"""
Load and use trained ImageRouter models for inference.

Usage:
    python load_model.py --model_dir ./models --timestamp 20240128_123456
    python load_model.py --checkpoint ./checkpoints/final_checkpoint_20240128_123456.pt
"""

import argparse
import os
import torch
import torch.nn as nn
from router_models import SimpleCNN, RouterModel
from router_system import RouterSystem
from utils import get_imagenet_dataloaders
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime


def load_from_checkpoint(checkpoint_path, device='cuda'):
    """Load model from a checkpoint file (includes training state)."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    N = ckpt['branches']
    num_classes = 100  # CIFAR-100

    # Reconstruct branches (need to know architecture)
    # This is a limitation - we need to know the architecture
    # In practice, you'd save this info in the checkpoint
    branches = []
    for i in range(N):
        branch = SimpleCNN(num_classes=num_classes, base_channels=32, num_layers=3)
        branch.load_state_dict(ckpt['branch_states'][i])
        branch = branch.to(device)
        branch.eval()
        branches.append(branch)

    # Load router
    router = RouterModel(N).to(device)
    router.load_state_dict(ckpt['router_state'])
    router.eval()

    # Create network
    costs = ckpt['costs'].to(device)
    network = RouterSystem(router, branches, costs)
    network.eval()

    print(f"✓ Loaded model from step {ckpt['step']}")
    print(f"  Branches: {N}")
    print(f"  Costs: {costs.tolist()}")

    return network, branches, router


def load_from_weights(models_dir, timestamp, device='cuda'):
    """Load model from individual weight files (inference only)."""
    print(f"Loading models from: {models_dir}")
    print(f"Timestamp: {timestamp}")

    # Load model info
    info_path = os.path.join(models_dir, f"model_info_{timestamp}.pt")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Model info not found: {info_path}")

    model_info = torch.load(info_path, map_location=device)
    N = model_info['num_branches']
    branch_params = model_info['branch_params']
    num_classes = model_info['num_classes']

    print(f"  Branches: {N}")
    print(f"  Classes: {num_classes}")
    print(f"  Training steps: {model_info['total_steps']}")

    # Load branches
    branches = []
    for i in range(N):
        ch, layers, cost = branch_params[i]
        branch = SimpleCNN(num_classes=num_classes, base_channels=ch, num_layers=layers)

        path = os.path.join(models_dir, f"branch_{i}_{timestamp}.pth")
        branch.load_state_dict(torch.load(path, map_location=device))
        branch = branch.to(device)
        branch.eval()
        branches.append(branch)
        print(f"  ✓ Branch {i}: {ch} channels, {layers} layers")

    # Load router
    router = RouterModel(N).to(device)
    router_path = os.path.join(models_dir, f"router_{timestamp}.pth")
    router.load_state_dict(torch.load(router_path, map_location=device))
    router.eval()
    print(f"  ✓ Router loaded")

    # Create network
    costs = torch.tensor([cost for _, _, cost in branch_params], device=device)
    network = RouterSystem(router, branches, costs)
    network.eval()

    return network, branches, router, model_info


def evaluate_on_cifar100(network, branches, device='cuda', batch_size=128, num_workers=2):
    """Evaluate the model on CIFAR-100 test set."""
    print("\n" + "=" * 60)
    print("Evaluating on CIFAR-100 test set...")
    print("=" * 60)

    # Load CIFAR-100 test data
    _, val_loader = get_imagenet_dataloaders(
        './data',
        batch_size=batch_size,
        num_workers=num_workers,
        subset_size=None,
        download=True
    )

    N = len(branches)
    network.eval()
    for b in branches:
        b.eval()

    # Overall statistics
    total_correct = 0
    total_samples = 0
    routing_dist = [0] * N

    # Per-branch statistics
    branch_correct = [0] * N
    branch_total = [0] * N

    print(f"Total batches: {len(val_loader)}")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Get routing decisions
            routes = torch.argmax(network.router(x), dim=-1)

            # Get predictions for each sample
            logits = torch.empty((x.size(0), 100), device=device)

            for i in range(N):
                mask = (routes == i)
                if mask.any():
                    x_i = x[mask]
                    y_i = y[mask]
                    out_i = branches[i](x_i)
                    logits[mask] = out_i

                    # Track per-branch accuracy
                    routing_dist[i] += int(mask.sum().item())
                    branch_correct[i] += int((out_i.argmax(1) == y_i).sum().item())
                    branch_total[i] += int(mask.sum().item())

            # Overall accuracy
            preds = logits.argmax(1)
            total_correct += int((preds == y).sum().item())
            total_samples += int(x.size(0))

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches...")

    # Print results
    overall_acc = total_correct / total_samples
    print(f"\n{'=' * 60}")
    print(f"Overall Accuracy: {overall_acc:.4f} ({total_correct}/{total_samples})")
    print(f"{'=' * 60}")

    print("\nRouting Distribution:")
    for i in range(N):
        pct = routing_dist[i] / total_samples
        print(f"  Branch {i}: {pct:.2%} ({routing_dist[i]} samples)")

    print("\nPer-Branch Accuracy (on routed samples):")
    for i in range(N):
        if branch_total[i] > 0:
            acc = branch_correct[i] / branch_total[i]
            print(f"  Branch {i}: {acc:.4f} ({branch_correct[i]}/{branch_total[i]})")
        else:
            print(f"  Branch {i}: N/A (no samples routed)")

    return overall_acc


def predict_single_image(network, branches, image_tensor, device='cuda'):
    """
    Predict class for a single image.

    Args:
        network: RouterSystem
        branches: List of branch models
        image_tensor: Tensor of shape (3, 32, 32) or (1, 3, 32, 32)
        device: Device to run on

    Returns:
        predicted_class: int
        branch_used: int
        confidence: float
    """
    network.eval()
    for b in branches:
        b.eval()

    # Add batch dimension if needed
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Get routing decision
        route = torch.argmax(network.router(image_tensor), dim=-1).item()

        # Get prediction from selected branch
        logits = branches[route](image_tensor)
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_class = torch.max(probs, dim=-1)

        return int(pred_class.item()), route, float(confidence.item())


def visualize_sample_routing(network, branches, device='cuda', num_samples=16, save_path='sample_routing_analysis.png'):
    """
    Visualize sample images with their routing decisions and predictions.
    """
    print("\n" + "=" * 60)
    print("Generating sample routing visualization...")
    print("=" * 60)

    # Load CIFAR-100 test data
    _, val_loader = get_imagenet_dataloaders(
        './data',
        batch_size=num_samples,
        num_workers=2,
        subset_size=None,
        download=True
    )

    # CIFAR-100 classes
    classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
        'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
        'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
        'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
        'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
        'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

    # Get one batch of samples
    x, y = next(iter(val_loader))
    x = x.to(device)
    y = y.to(device)

    N = len(branches)
    network.eval()
    for b in branches:
        b.eval()

    with torch.no_grad():
        # Get routing decisions
        routes = torch.argmax(network.router(x), dim=-1)

        # Get predictions for each sample
        preds = []
        confidences = []
        for i in range(x.size(0)):
            route = routes[i].item()
            logits = branches[route](x[i:i+1])
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            preds.append(pred.item())
            confidences.append(conf.item())

    # Denormalize images for display
    mean = torch.tensor([0.5071, 0.4867, 0.4408], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761], device=device).view(1, 3, 1, 1)
    x_display = x * std + mean
    x_display = torch.clamp(x_display, 0, 1)

    # Create visualization
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle('Sample Routing Decisions', fontsize=16, fontweight='bold')

    for idx in range(min(num_samples, rows * cols)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Display image
        img = x_display[idx].cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')

        # Add routing info
        true_label = classes[y[idx].item()]
        pred_label = classes[preds[idx]]
        route = routes[idx].item()
        conf = confidences[idx]

        correct = "✓" if y[idx].item() == preds[idx] else "✗"
        color = 'green' if y[idx].item() == preds[idx] else 'red'

        title = f"{correct} Route: Branch {route}\n"
        title += f"True: {true_label}\n"
        title += f"Pred: {pred_label}\n"
        title += f"Conf: {conf:.2f}"

        ax.set_title(title, fontsize=8, color=color, pad=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {save_path}")
    plt.close()


def analyze_time_savings(network, branches, router, costs, device='cuda', num_batches=50, batch_size=128):
    """
    Benchmark inference time: router system vs running all branches.
    """
    print("\n" + "=" * 60)
    print("Analyzing Time Savings...")
    print("=" * 60)

    # Load CIFAR-100 test data
    _, val_loader = get_imagenet_dataloaders(
        './data',
        batch_size=batch_size,
        num_workers=2,
        subset_size=None,
        download=True
    )

    N = len(branches)
    network.eval()
    router.eval()
    for b in branches:
        b.eval()

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for i, (x, _) in enumerate(val_loader):
            if i >= 5:
                break
            x = x.to(device)
            _ = router(x)
            for branch in branches:
                _ = branch(x)

    # Benchmark router system
    print(f"Benchmarking router system ({num_batches} batches)...")
    router_times = []
    with torch.no_grad():
        for i, (x, _) in enumerate(val_loader):
            if i >= num_batches:
                break
            x = x.to(device)

            torch.cuda.synchronize() if device.startswith('cuda') else None
            start = time.time()

            # Router decision + branch inference
            routes = torch.argmax(router(x), dim=-1)
            for branch_idx in range(N):
                mask = (routes == branch_idx)
                if mask.any():
                    _ = branches[branch_idx](x[mask])

            torch.cuda.synchronize() if device.startswith('cuda') else None
            end = time.time()
            router_times.append(end - start)

    # Benchmark all branches (no router)
    print(f"Benchmarking all branches ({num_batches} batches)...")
    all_branch_times = []
    with torch.no_grad():
        for i, (x, _) in enumerate(val_loader):
            if i >= num_batches:
                break
            x = x.to(device)

            torch.cuda.synchronize() if device.startswith('cuda') else None
            start = time.time()

            # Run all branches
            for branch in branches:
                _ = branch(x)

            torch.cuda.synchronize() if device.startswith('cuda') else None
            end = time.time()
            all_branch_times.append(end - start)

    # Benchmark individual branches
    print("Benchmarking individual branches...")
    branch_times = []
    for branch_idx, branch in enumerate(branches):
        times = []
        with torch.no_grad():
            for i, (x, _) in enumerate(val_loader):
                if i >= num_batches:
                    break
                x = x.to(device)

                torch.cuda.synchronize() if device.startswith('cuda') else None
                start = time.time()
                _ = branch(x)
                torch.cuda.synchronize() if device.startswith('cuda') else None
                end = time.time()
                times.append(end - start)
        branch_times.append(np.mean(times))

    # Calculate statistics
    router_avg = np.mean(router_times) * 1000  # Convert to ms
    all_branch_avg = np.mean(all_branch_times) * 1000
    speedup = all_branch_avg / router_avg
    time_saved = all_branch_avg - router_avg

    # Print results
    print(f"\n{'=' * 60}")
    print("Time Analysis Results:")
    print(f"{'=' * 60}")
    print(f"Router system:     {router_avg:.2f} ms/batch")
    print(f"All branches:      {all_branch_avg:.2f} ms/batch")
    print(f"Speedup:           {speedup:.2f}x")
    print(f"Time saved:        {time_saved:.2f} ms/batch ({time_saved/all_branch_avg*100:.1f}%)")

    print(f"\nIndividual branch times:")
    for i, t in enumerate(branch_times):
        cost = costs[i].item() if hasattr(costs, '__getitem__') else costs
        print(f"  Branch {i} (cost={cost:.2f}): {t*1000:.2f} ms/batch")

    # Throughput
    samples_per_sec_router = (batch_size * num_batches) / sum(router_times)
    samples_per_sec_all = (batch_size * num_batches) / sum(all_branch_times)

    print(f"\nThroughput:")
    print(f"  Router system: {samples_per_sec_router:.1f} samples/sec")
    print(f"  All branches:  {samples_per_sec_all:.1f} samples/sec")

    return {
        'router_time_ms': router_avg,
        'all_branches_time_ms': all_branch_avg,
        'speedup': speedup,
        'time_saved_ms': time_saved,
        'branch_times_ms': [t*1000 for t in branch_times]
    }


def analyze_accuracy_tradeoffs(network, branches, router, costs, device='cuda', num_batches=None):
    """
    Compare router accuracy vs oracle (always picking best branch) vs individual branches.
    """
    print("\n" + "=" * 60)
    print("Analyzing Accuracy Tradeoffs...")
    print("=" * 60)

    # Load CIFAR-100 test data
    _, val_loader = get_imagenet_dataloaders(
        './data',
        batch_size=128,
        num_workers=2,
        subset_size=None,
        download=True
    )

    N = len(branches)
    network.eval()
    router.eval()
    for b in branches:
        b.eval()

    # Statistics
    total_samples = 0
    router_correct = 0
    oracle_correct = 0  # Best possible with perfect routing
    branch_correct = [0] * N  # Each branch on all samples
    routing_dist = [0] * N

    # Track when router disagrees with oracle
    oracle_disagreements = 0
    router_correct_oracle_wrong = 0

    print("Evaluating...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            if num_batches and batch_idx >= num_batches:
                break

            x = x.to(device)
            y = y.to(device)

            # Get router decisions
            routes = torch.argmax(router(x), dim=-1)

            # Get predictions from all branches
            all_logits = []
            for branch in branches:
                logits = branch(x)
                all_logits.append(logits)
            all_logits = torch.stack(all_logits, dim=0)  # (N, B, C)

            # Router accuracy (use routed branch)
            router_preds = torch.empty(x.size(0), dtype=torch.long, device=device)
            for i in range(N):
                mask = (routes == i)
                if mask.any():
                    router_preds[mask] = all_logits[i, mask].argmax(dim=-1)
                    routing_dist[i] += mask.sum().item()

            router_correct += (router_preds == y).sum().item()

            # Oracle accuracy (always pick branch that would be correct)
            # For each sample, check which branch gives correct prediction
            all_preds = all_logits.argmax(dim=-1)  # (N, B)
            correct_by_branch = (all_preds == y.unsqueeze(0))  # (N, B)

            # For each sample, oracle picks any branch that's correct (if any exists)
            oracle_correct_batch = correct_by_branch.any(dim=0)  # (B,)
            oracle_correct += oracle_correct_batch.sum().item()

            # Individual branch accuracy
            for i in range(N):
                branch_correct[i] += correct_by_branch[i].sum().item()

            # Track disagreements
            for sample_idx in range(x.size(0)):
                route = routes[sample_idx].item()
                router_correct_sample = (router_preds[sample_idx] == y[sample_idx]).item()
                oracle_correct_sample = oracle_correct_batch[sample_idx].item()

                if router_correct_sample != oracle_correct_sample:
                    oracle_disagreements += 1
                    if router_correct_sample and not oracle_correct_sample:
                        router_correct_oracle_wrong += 1

            total_samples += x.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches...")

    # Calculate accuracies
    router_acc = router_correct / total_samples
    oracle_acc = oracle_correct / total_samples
    branch_accs = [c / total_samples for c in branch_correct]

    # Print results
    print(f"\n{'=' * 60}")
    print("Accuracy Tradeoff Results:")
    print(f"{'=' * 60}")
    print(f"Router accuracy:       {router_acc:.4f} ({router_correct}/{total_samples})")
    print(f"Oracle accuracy:       {oracle_acc:.4f} ({oracle_correct}/{total_samples})")
    print(f"Accuracy gap:          {(oracle_acc - router_acc):.4f} ({(oracle_acc - router_acc)*100:.2f}%)")

    print(f"\nIndividual branch accuracies (if used for all samples):")
    for i in range(N):
        cost = costs[i].item() if hasattr(costs, '__getitem__') else costs
        print(f"  Branch {i} (cost={cost:.2f}): {branch_accs[i]:.4f}")

    print(f"\nRouter behavior:")
    print(f"  Oracle disagreements: {oracle_disagreements} ({oracle_disagreements/total_samples*100:.2f}%)")
    print(f"  Router correct when oracle wrong: {router_correct_oracle_wrong}")

    print(f"\nRouting distribution:")
    for i in range(N):
        pct = routing_dist[i] / total_samples
        print(f"  Branch {i}: {pct:.2%} ({routing_dist[i]} samples)")

    return {
        'router_accuracy': router_acc,
        'oracle_accuracy': oracle_acc,
        'accuracy_gap': oracle_acc - router_acc,
        'branch_accuracies': branch_accs,
        'routing_distribution': [d / total_samples for d in routing_dist]
    }


def main():
    parser = argparse.ArgumentParser(description='Load and evaluate trained ImageRouter model')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory with model weights')
    parser.add_argument('--timestamp', type=str, help='Model timestamp (required if using --model_dir)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on CIFAR-100')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--analyze', action='store_true', help='Run full analysis (routing viz, time, accuracy tradeoffs)')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample routing decisions')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark inference time')
    parser.add_argument('--tradeoffs', action='store_true', help='Analyze accuracy tradeoffs')
    parser.add_argument('--save_dir', type=str, default='./analysis', help='Directory to save analysis results')

    args = parser.parse_args()

    print("=" * 60)
    print("ImageRouter Model Loader")
    print("=" * 60)
    print(f"Device: {args.device}")
    print()

    # Load model
    if args.checkpoint:
        network, branches, router = load_from_checkpoint(args.checkpoint, args.device)
        model_info = None
        costs = network.costs
    elif args.model_dir and args.timestamp:
        network, branches, router, model_info = load_from_weights(
            args.model_dir, args.timestamp, args.device
        )
        costs = network.costs
    else:
        parser.error("Must specify either --checkpoint or both --model_dir and --timestamp")

    # Create save directory
    if args.analyze or args.visualize or args.benchmark or args.tradeoffs:
        os.makedirs(args.save_dir, exist_ok=True)

    # Evaluate if requested
    if args.evaluate:
        accuracy = evaluate_on_cifar100(
            network, branches, args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        if model_info:
            print(f"\nTraining info:")
            print(f"  Timestamp: {model_info['timestamp']}")
            print(f"  Total steps: {model_info['total_steps']}")

    # Run analysis if requested
    if args.analyze or args.visualize:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(args.save_dir, f'sample_routing_{timestamp}.png')
        visualize_sample_routing(network, branches, args.device, save_path=viz_path)

    if args.analyze or args.benchmark:
        time_stats = analyze_time_savings(network, branches, router, costs, args.device)

    if args.analyze or args.tradeoffs:
        acc_stats = analyze_accuracy_tradeoffs(network, branches, router, costs, args.device)

    print("\n" + "=" * 60)
    print("✓ Model loaded successfully!")
    print("=" * 60)

    # Example: predict on a random image (only if no analysis was run)
    if not (args.analyze or args.visualize or args.benchmark or args.tradeoffs):
        print("\nExample prediction on a random image:")
        dummy_img = torch.randn(1, 3, 32, 32)
        pred_class, branch, confidence = predict_single_image(network, branches, dummy_img, args.device)
        print(f"  Predicted class: {pred_class}")
        print(f"  Branch used: {branch}")
        print(f"  Confidence: {confidence:.4f}")
        print("\nTip: Use --analyze to run full analysis (routing visualization, time benchmarks, accuracy tradeoffs)")
        print("     Or use --visualize, --benchmark, --tradeoffs individually")


if __name__ == '__main__':
    main()
