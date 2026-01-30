#!/usr/bin/env python3
"""
Export new 5-branch models with updated architecture.

Usage:
    python3 export_new_models.py
"""

import torch
from pathlib import Path
import sys

# Import model classes from router_models
sys.path.insert(0, str(Path(__file__).parent))
from router_models import SimpleCNN, RouterModel


# Branch configurations: (base_channels, num_layers)
BRANCH_CONFIGS = [
    (64, 4),   # Branch 0 → 512
    (88, 4),   # Branch 1 → 704
    (64, 5),   # Branch 2 → 1024
    (80, 5),   # Branch 3 → 1280
    (96, 5),   # Branch 4 → 1536
]

NUM_BRANCHES = 5
NUM_CLASSES = 100
TIMESTAMP = "20260130_075616"


def export_models():
    """Export the new 5-branch models."""
    print("=" * 60)
    print("Exporting New 5-Branch Models")
    print("=" * 60)
    print(f"\nTimestamp: {TIMESTAMP}")
    print(f"Branches: {NUM_BRANCHES}")
    print()

    device = 'cpu'
    output_dir = Path('backend/models')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export router
    print("Exporting router...")
    router_path_src = f"models/router_{TIMESTAMP}.pth"

    if not Path(router_path_src).exists():
        print(f"ERROR: Router not found: {router_path_src}")
        return False

    # Use RouterModel from router_models.py (with dropout)
    router = RouterModel(NUM_BRANCHES, drop2d=0.1, drop=0.2)
    router.load_state_dict(torch.load(router_path_src, map_location=device, weights_only=True))
    router.eval()

    router_path_dst = output_dir / 'router.pth'
    torch.save(router, router_path_dst)
    print(f"  ✓ Saved to: {router_path_dst}")

    # Export branches
    print(f"\nExporting {NUM_BRANCHES} branches...")
    for i in range(NUM_BRANCHES):
        channels, layers = BRANCH_CONFIGS[i]

        # Use SimpleCNN from router_models.py (with dropout)
        branch = SimpleCNN(
            num_classes=NUM_CLASSES,
            base_channels=channels,
            num_layers=layers,
            drop2d=0.1,
            drop=0.3
        )

        branch_path_src = f"models/branch_{i}_{TIMESTAMP}.pth"

        if not Path(branch_path_src).exists():
            print(f"ERROR: Branch {i} not found: {branch_path_src}")
            return False

        branch.load_state_dict(torch.load(branch_path_src, map_location=device, weights_only=True))
        branch.eval()

        branch_path_dst = output_dir / f'branch_{i}.pth'
        torch.save(branch, branch_path_dst)

        # Calculate final channels
        final_ch = channels * (2 ** (layers - 1))
        print(f"  ✓ Branch {i} ({channels} ch, {layers} layers → {final_ch}) saved to: {branch_path_dst}")

    print("\n" + "=" * 60)
    print("✓ Export complete!")
    print("=" * 60)
    print(f"\nModels saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. python3 verify_setup.py")
    print("  2. cd backend && python3 run.py")
    print("  3. cd frontend && npm run dev")

    return True


if __name__ == '__main__':
    success = export_models()
    exit(0 if success else 1)
