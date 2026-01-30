#!/usr/bin/env python3
"""
Direct model export without requiring model_info file.
Specify branch architectures manually.

Usage:
    python3 export_models_direct.py --model_dir ./models --timestamp TIMESTAMP
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Import model classes
try:
    from router_models import SimpleCNN, RouterModel
except ImportError:
    print("ERROR: Could not import router_models.")
    print("Make sure router_models.py is in the same directory.")
    sys.exit(1)


# CONFIGURE YOUR BRANCH ARCHITECTURES HERE
BRANCH_CONFIGS = [
    # (base_channels, num_layers)
    (32, 3),   # Branch 0
    (48, 3),   # Branch 1
    (64, 3),   # Branch 2
    (96, 3),   # Branch 3
]

NUM_BRANCHES = 4
NUM_CLASSES = 100


def export_models(model_dir, timestamp, output_dir='backend/models'):
    """Export models directly without model_info file."""
    print("=" * 60)
    print("Direct Model Exporter")
    print("=" * 60)
    print(f"\nModel directory: {model_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Branches: {NUM_BRANCHES}")
    print(f"Classes: {NUM_CLASSES}")
    print()

    device = 'cpu'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export router
    print("Exporting router...")
    router_path_src = os.path.join(model_dir, f"router_{timestamp}.pth")

    if not os.path.exists(router_path_src):
        print(f"ERROR: Router not found: {router_path_src}")
        return False

    router = RouterModel(NUM_BRANCHES)
    router.load_state_dict(torch.load(router_path_src, map_location=device, weights_only=True))
    router.eval()

    router_path_dst = output_path / 'router.pth'
    torch.save(router, router_path_dst)
    print(f"  ✓ Saved to: {router_path_dst}")

    # Export branches
    print(f"\nExporting {NUM_BRANCHES} branches...")
    for i in range(NUM_BRANCHES):
        if i >= len(BRANCH_CONFIGS):
            print(f"ERROR: No config for branch {i}. Update BRANCH_CONFIGS in this script.")
            return False

        channels, layers = BRANCH_CONFIGS[i]
        branch = SimpleCNN(num_classes=NUM_CLASSES, base_channels=channels, num_layers=layers)

        branch_path_src = os.path.join(model_dir, f"branch_{i}_{timestamp}.pth")

        if not os.path.exists(branch_path_src):
            print(f"ERROR: Branch {i} not found: {branch_path_src}")
            return False

        branch.load_state_dict(torch.load(branch_path_src, map_location=device, weights_only=True))
        branch.eval()

        branch_path_dst = output_path / f'branch_{i}.pth'
        torch.save(branch, branch_path_dst)
        print(f"  ✓ Branch {i} ({channels} ch, {layers} layers) saved to: {branch_path_dst}")

    print("\n" + "=" * 60)
    print("✓ Export complete!")
    print("=" * 60)
    print(f"\nModels saved to: {output_path}")
    print("\nNext steps:")
    print("  1. python3 verify_setup.py")
    print("  2. cd backend && python3 run.py")
    print("  3. cd frontend && npm run dev")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Export models directly without model_info file'
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory with model weight files')
    parser.add_argument('--timestamp', type=str, required=True,
                        help='Model timestamp')
    parser.add_argument('--output_dir', type=str, default='backend/models',
                        help='Output directory for exported models')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BRANCH CONFIGURATIONS")
    print("=" * 60)
    for i, (ch, layers) in enumerate(BRANCH_CONFIGS):
        print(f"  Branch {i}: {ch} channels, {layers} layers")
    print()
    print("If these don't match your trained models, edit BRANCH_CONFIGS")
    print("in this script before running.")
    print("=" * 60)
    print()

    success = export_models(args.model_dir, args.timestamp, args.output_dir)

    if not success:
        print("\n✗ Export failed. Please check the error messages above.")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
