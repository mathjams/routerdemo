#!/usr/bin/env python3
"""
Export trained router and branch models for the demo application.

This script loads your trained models and exports them in a format
compatible with the FastAPI demo backend.

Usage:
    python export_models_for_demo.py --model_dir ./models --timestamp 20240128_123456
    python export_models_for_demo.py --checkpoint ./checkpoints/final_checkpoint.pt
"""

import argparse
import os
import torch
from pathlib import Path


def export_from_checkpoint(checkpoint_path, output_dir='backend/models'):
    """Export models from a checkpoint file."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Import your model classes
    try:
        from router_models import SimpleCNN, RouterModel
    except ImportError:
        print("ERROR: Could not import router_models.")
        print("Make sure router_models.py is in the same directory.")
        return False

    device = 'cpu'  # Always export on CPU for compatibility
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    N = ckpt['branches']
    num_classes = 100  # CIFAR-100

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export router
    print(f"\nExporting router...")
    router = RouterModel(N)
    router.load_state_dict(ckpt['router_state'])
    router.eval()

    router_path = output_path / 'router.pth'
    torch.save(router, router_path)
    print(f"  ✓ Saved to: {router_path}")

    # Export branches
    print(f"\nExporting {N} branches...")
    for i in range(N):
        branch = SimpleCNN(num_classes=num_classes, base_channels=32, num_layers=3)
        branch.load_state_dict(ckpt['branch_states'][i])
        branch.eval()

        branch_path = output_path / f'branch_{i}.pth'
        torch.save(branch, branch_path)
        print(f"  ✓ Branch {i} saved to: {branch_path}")

    print(f"\n{'='*60}")
    print("✓ Export complete!")
    print(f"{'='*60}")
    print(f"Models saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Update backend/config.py to set BRANCH_PATHS for {N} branches")
    print(f"2. Run: cd backend && python run.py")
    print(f"3. Run: cd frontend && npm run dev")

    return True


def export_from_weights(models_dir, timestamp, output_dir='backend/models'):
    """Export models from individual weight files."""
    print(f"Loading models from: {models_dir}")
    print(f"Timestamp: {timestamp}")

    # Import your model classes
    try:
        from router_models import SimpleCNN, RouterModel
    except ImportError:
        print("ERROR: Could not import router_models.")
        print("Make sure router_models.py is in the same directory.")
        return False

    device = 'cpu'  # Always export on CPU for compatibility

    # Load model info
    info_path = os.path.join(models_dir, f"model_info_{timestamp}.pt")
    if not os.path.exists(info_path):
        print(f"ERROR: Model info not found: {info_path}")
        return False

    model_info = torch.load(info_path, map_location=device, weights_only=False)
    N = model_info['num_branches']
    branch_params = model_info['branch_params']
    num_classes = model_info['num_classes']

    print(f"  Branches: {N}")
    print(f"  Classes: {num_classes}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export router
    print(f"\nExporting router...")
    router = RouterModel(N)
    router_path_src = os.path.join(models_dir, f"router_{timestamp}.pth")
    router.load_state_dict(torch.load(router_path_src, map_location=device, weights_only=True))
    router.eval()

    router_path_dst = output_path / 'router.pth'
    torch.save(router, router_path_dst)
    print(f"  ✓ Saved to: {router_path_dst}")

    # Export branches
    print(f"\nExporting {N} branches...")
    for i in range(N):
        ch, layers, cost = branch_params[i]
        branch = SimpleCNN(num_classes=num_classes, base_channels=ch, num_layers=layers)

        branch_path_src = os.path.join(models_dir, f"branch_{i}_{timestamp}.pth")
        branch.load_state_dict(torch.load(branch_path_src, map_location=device, weights_only=True))
        branch.eval()

        branch_path_dst = output_path / f'branch_{i}.pth'
        torch.save(branch, branch_path_dst)
        print(f"  ✓ Branch {i} ({ch} channels, {layers} layers) saved to: {branch_path_dst}")

    print(f"\n{'='*60}")
    print("✓ Export complete!")
    print(f"{'='*60}")
    print(f"Models saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Update backend/config.py to set BRANCH_PATHS for {N} branches")
    print(f"2. Run: cd backend && python run.py")
    print(f"3. Run: cd frontend && npm run dev")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Export trained models for the demo application'
    )
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Directory with model weights')
    parser.add_argument('--timestamp', type=str,
                        help='Model timestamp (required if using --model_dir)')
    parser.add_argument('--output_dir', type=str, default='backend/models',
                        help='Output directory for exported models')

    args = parser.parse_args()

    print("=" * 60)
    print("Model Exporter for Demo Application")
    print("=" * 60)
    print()

    # Export models
    success = False
    if args.checkpoint:
        success = export_from_checkpoint(args.checkpoint, args.output_dir)
    elif args.model_dir and args.timestamp:
        success = export_from_weights(args.model_dir, args.timestamp, args.output_dir)
    else:
        parser.error("Must specify either --checkpoint or both --model_dir and --timestamp")

    if not success:
        print("\n✗ Export failed. Please check the error messages above.")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
