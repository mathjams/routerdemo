#!/usr/bin/env python3
"""
Verify that your adaptive router demo is set up correctly.
Run this before starting the servers to catch common issues.

Usage:
    python verify_setup.py
"""

import sys
from pathlib import Path


def check_file(path, description):
    """Check if a file exists."""
    if path.exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {path}")
        return False


def check_directory(path, description):
    """Check if a directory exists."""
    if path.is_dir():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {path}")
        return False


def main():
    print("=" * 60)
    print("Adaptive Router Demo - Setup Verification")
    print("=" * 60)
    print()

    issues = []
    warnings = []

    # Check Python dependencies
    print("Checking Python dependencies...")
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch NOT installed")
        issues.append("Install PyTorch: pip install torch torchvision")

    try:
        import fastapi
        print(f"✓ FastAPI installed: {fastapi.__version__}")
    except ImportError:
        print("✗ FastAPI NOT installed")
        issues.append("Install FastAPI: pip install fastapi")

    try:
        import PIL
        print(f"✓ Pillow installed: {PIL.__version__}")
    except ImportError:
        print("✗ Pillow NOT installed")
        issues.append("Install Pillow: pip install Pillow")

    try:
        from fvcore.nn import FlopCountAnalysis
        print("✓ fvcore installed (FLOPs calculation available)")
    except ImportError:
        print("⚠ fvcore NOT installed (FLOPs calculation will be disabled)")
        warnings.append("Install fvcore for FLOPs: pip install fvcore")

    print()

    # Check backend structure
    print("Checking backend structure...")
    backend_dir = Path("backend")
    if not check_directory(backend_dir, "Backend directory"):
        issues.append("Backend directory not found")
        return 1

    if not check_file(backend_dir / "config.py", "Backend config"):
        issues.append("Backend config.py not found")

    if not check_file(backend_dir / "run.py", "Backend run script"):
        issues.append("Backend run.py not found")

    if not check_directory(backend_dir / "app", "Backend app directory"):
        issues.append("Backend app directory not found")

    print()

    # Check model files
    print("Checking model files...")
    models_dir = backend_dir / "models"
    if not check_directory(models_dir, "Models directory"):
        issues.append("Models directory not found")
    else:
        router_path = models_dir / "router.pth"
        if not check_file(router_path, "Router model"):
            issues.append("Router model not found - run export_models_for_demo.py")

        # Check for branch models
        branch_files = list(models_dir.glob("branch_*.pth"))
        if len(branch_files) == 0:
            print(f"✗ No branch models found in {models_dir}")
            issues.append("Branch models not found - run export_models_for_demo.py")
        else:
            print(f"✓ Found {len(branch_files)} branch model(s)")
            for bf in sorted(branch_files):
                print(f"  - {bf.name}")

    print()

    # Check config.py settings
    print("Checking configuration...")
    try:
        sys.path.insert(0, str(backend_dir))
        import config

        num_branches_config = len(config.BRANCH_PATHS)
        print(f"✓ Config loaded: {num_branches_config} branches configured")

        # Check if configured branches match actual files
        if len(branch_files) > 0 and len(branch_files) != num_branches_config:
            print(f"⚠ WARNING: Found {len(branch_files)} branch files but config has {num_branches_config}")
            warnings.append(f"Update BRANCH_PATHS in config.py to have {len(branch_files)} branches")

    except Exception as e:
        print(f"✗ Error loading config: {e}")
        issues.append("Fix config.py errors")

    print()

    # Check frontend structure
    print("Checking frontend structure...")
    frontend_dir = Path("frontend")
    if not check_directory(frontend_dir, "Frontend directory"):
        warnings.append("Frontend directory not found")
    else:
        if not check_file(frontend_dir / "package.json", "Frontend package.json"):
            issues.append("Frontend package.json not found")

        node_modules = frontend_dir / "node_modules"
        if not node_modules.exists():
            print(f"⚠ Node modules not installed")
            warnings.append("Run: cd frontend && npm install")
        else:
            print(f"✓ Node modules installed")

    print()

    # Check export script
    print("Checking export script...")
    export_script = Path("export_models_for_demo.py")
    if not check_file(export_script, "Export script"):
        issues.append("Export script not found")

    print()

    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    if len(issues) == 0 and len(warnings) == 0:
        print("✓ All checks passed! You're ready to go.")
        print()
        print("Next steps:")
        print("  1. cd backend && python run.py")
        print("  2. cd frontend && npm run dev  (in a new terminal)")
        print("  3. Open http://localhost:5173")
        return 0
    else:
        if len(issues) > 0:
            print(f"\n✗ {len(issues)} CRITICAL ISSUE(S) FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

        if len(warnings) > 0:
            print(f"\n⚠ {len(warnings)} WARNING(S):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")

        print()
        print("Please fix the issues above before starting the demo.")
        return 1


if __name__ == '__main__':
    exit(main())
