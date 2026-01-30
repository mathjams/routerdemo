#!/usr/bin/env python3
"""
Auto-detect architecture and export models without model_info.
Infers base_channels and num_layers from weight shapes.

Usage:
    python3 export_models_auto.py --model_dir ./models --timestamp TIMESTAMP
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


def detect_branch_architecture(state_dict):
    """
    Detect SimpleCNN architecture from state_dict.

    Returns:
        (base_channels, num_layers) or None if can't detect
    """
    # SimpleCNN first conv layer: features.0.weight has shape [base_channels, 3, 7, 7]
    if 'features.0.weight' in state_dict:
        first_conv = state_dict['features.0.weight']
        base_channels = first_conv.shape[0]

        # Count conv layers to determine num_layers
        conv_count = 0
        for key in state_dict.keys():
            if 'features.' in key and '.weight' in key and 'conv' not in key.lower():
                # Count Conv2d layers (features.X.weight where X is int)
                try:
                    layer_num = int(key.split('.')[1])
                    if 'weight' in key and state_dict[key].dim() == 4:  # Conv2d has 4D weights
                        conv_count += 1
                except (ValueError, IndexError):
                    pass

        # SimpleCNN structure: 1 initial conv + (num_layers-1) additional convs
        # First block: Conv + BN + ReLU + MaxPool = features.0-3
        # Each additional layer: Conv + BN + ReLU = 3 layers each
        # So total features layers = 4 + (num_layers-1) * 3

        # Count non-BN, non-activation layers
        total_layers = len([k for k in state_dict.keys() if 'features.' in k and '.weight' in k])

        # Estimate num_layers (this is approximate)
        if total_layers <= 7:  # 4 (first block) + 3 (one more layer)
            num_layers = 2
        elif total_layers <= 10:  # 4 + 6 (two more layers)
            num_layers = 3
        elif total_layers <= 13:  # 4 + 9 (three more layers)
            num_layers = 4
        else:
            num_layers = 5

        return base_channels, num_layers

    return None


def try_load_branch(state_dict, num_classes=100):
    """
    Try different architectures until one successfully loads the state_dict.

    Returns:
        (loaded_model, base_channels, num_layers) or (None, None, None)
    """
    # Try to detect first
    detected = detect_branch_architecture(state_dict)
    if detected:
        base_channels, num_layers = detected
        try:
            model = SimpleCNN(num_classes=num_classes, base_channels=base_channels, num_layers=num_layers)
            model.load_state_dict(state_dict)
            return model, base_channels, num_layers
        except Exception as e:
            print(f"    Detection failed ({base_channels}ch, {num_layers}L): {e}")

    # If detection fails, try common configurations
    common_configs = [
        (32, 2), (32, 3), (32, 4),
        (48, 2), (48, 3), (48, 4),
        (64, 2), (64, 3), (64, 4),
        (96, 2), (96, 3), (96, 4),
        (128, 2), (128, 3), (128, 4),
    ]

    for base_channels, num_layers in common_configs:
        try:
            model = SimpleCNN(num_classes=num_classes, base_channels=base_channels, num_layers=num_layers)
            model.load_state_dict(state_dict)
            return model, base_channels, num_layers
        except Exception:
            pass

    return None, None, None


def export_models_auto(model_dir, timestamp, output_dir='backend/models', num_classes=100):
    """Export models with automatic architecture detection."""
    print("=" * 60)
    print("Automatic Model Exporter")
    print("=" * 60)
    print(f"\nModel directory: {model_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Classes: {num_classes}")
    print(f"\nAuto-detecting architectures from model weights...")
    print("=" * 60)
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

    # Load router state_dict to count branches
    router_state = torch.load(router_path_src, map_location=device, weights_only=True)

    # Detect number of branches from router classifier
    if 'classifier.weight' in router_state:
        num_branches = router_state['classifier.weight'].shape[0]
        print(f"  Detected {num_branches} branches from router")
    else:
        print("  ERROR: Could not detect number of branches")
        return False

    router = RouterModel(num_branches)
    router.load_state_dict(router_state)
    router.eval()

    router_path_dst = output_path / 'router.pth'
    torch.save(router, router_path_dst)
    print(f"  ✓ Saved to: {router_path_dst}")

    # Export branches with auto-detection
    print(f"\nExporting {num_branches} branches (auto-detecting architectures)...")

    for i in range(num_branches):
        branch_path_src = os.path.join(model_dir, f"branch_{i}_{timestamp}.pth")

        if not os.path.exists(branch_path_src):
            print(f"ERROR: Branch {i} not found: {branch_path_src}")
            return False

        print(f"\n  Branch {i}:")
        print(f"    Loading from: {branch_path_src}")

        # Load state dict
        branch_state = torch.load(branch_path_src, map_location=device, weights_only=True)

        # Try to load with auto-detection
        print(f"    Auto-detecting architecture...")
        branch, base_channels, num_layers = try_load_branch(branch_state, num_classes)

        if branch is None:
            print(f"    ERROR: Could not detect architecture for branch {i}")
            print(f"    Please specify manually or check if the model file is corrupted")
            return False

        branch.eval()

        # Count parameters
        params = sum(p.numel() for p in branch.parameters())
        params_m = params / 1e6

        print(f"    ✓ Detected: {base_channels} channels, {num_layers} layers")
        print(f"    ✓ Parameters: {params_m:.2f}M")

        branch_path_dst = output_path / f'branch_{i}.pth'
        torch.save(branch, branch_path_dst)
        print(f"    ✓ Saved to: {branch_path_dst}")

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
        description='Export models with automatic architecture detection'
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory with model weight files')
    parser.add_argument('--timestamp', type=str, required=True,
                        help='Model timestamp')
    parser.add_argument('--output_dir', type=str, default='backend/models',
                        help='Output directory for exported models')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='Number of output classes (default: 100 for CIFAR-100)')

    args = parser.parse_args()

    success = export_models_auto(
        args.model_dir,
        args.timestamp,
        args.output_dir,
        args.num_classes
    )

    if not success:
        print("\n✗ Export failed. Please check the error messages above.")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
