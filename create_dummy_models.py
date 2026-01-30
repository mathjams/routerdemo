#!/usr/bin/env python3
"""
Create dummy models for testing the demo infrastructure.
Only use this for testing - replace with real models for actual use.

Usage:
    python3 create_dummy_models.py
"""

import sys
import torch
from pathlib import Path

# Import from router_models module (not __main__)
try:
    from router_models import SimpleCNN, RouterModel
except ImportError:
    print("ERROR: Could not import router_models.py")
    print("Make sure router_models.py exists in the current directory")
    sys.exit(1)


def main():
    print("=" * 60)
    print("Creating Dummy Models for Testing")
    print("=" * 60)
    print("\nWARNING: These are UNTRAINED models for testing only!")
    print("Replace with real trained models for actual use.\n")

    # Create output directory
    output_dir = Path("backend/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create router
    print("Creating router model...")
    num_branches = 4
    router = RouterModel(num_branches=num_branches)
    router.eval()
    router_path = output_dir / "router.pth"
    torch.save(router, router_path)
    print(f"✓ Saved router to {router_path}")

    # Create branches with varying sizes
    print(f"\nCreating {num_branches} branch models...")
    branch_configs = [
        (32, 2),   # Small: 32 channels, 2 layers
        (32, 3),   # Medium: 32 channels, 3 layers
        (48, 3),   # Large: 48 channels, 3 layers
        (64, 3),   # Largest: 64 channels, 3 layers
    ]

    for i, (channels, layers) in enumerate(branch_configs[:num_branches]):
        branch = SimpleCNN(num_classes=100, base_channels=channels, num_layers=layers)
        branch.eval()
        branch_path = output_dir / f"branch_{i}.pth"
        torch.save(branch, branch_path)
        print(f"✓ Saved branch {i} ({channels} ch, {layers} layers) to {branch_path}")

    # Test models work
    print("\nTesting models...")
    x = torch.randn(2, 3, 32, 32)

    router = torch.load(router_path, map_location='cpu', weights_only=False)
    router_out = router(x)
    print(f"✓ Router output shape: {router_out.shape} (expected: [2, {num_branches}])")

    branch = torch.load(output_dir / "branch_0.pth", map_location='cpu', weights_only=False)
    branch_out = branch(x)
    print(f"✓ Branch output shape: {branch_out.shape} (expected: [2, 100])")

    print("\n" + "=" * 60)
    print("✓ Dummy models created successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. python3 verify_setup.py  (should pass now)")
    print("  2. cd backend && python3 run.py")
    print("  3. cd frontend && npm run dev")
    print("\nNOTE: Replace dummy models with real trained models for actual use!")


if __name__ == '__main__':
    main()
