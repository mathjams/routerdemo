"""Utility functions for model analysis."""
import torch
import torch.nn as nn
from typing import Dict, Optional

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not available. FLOPs calculation will be disabled.")


def count_parameters(model: nn.Module) -> int:
    """Count the total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def calculate_flops(model: nn.Module, input_shape: tuple = (1, 3, 32, 32)) -> Optional[int]:
    """Calculate FLOPs for a model with given input shape.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, channels, height, width)

    Returns:
        Total FLOPs or None if fvcore is not available
    """
    if not FVCORE_AVAILABLE:
        return None

    try:
        model.eval()
        dummy_input = torch.randn(input_shape)
        flops = FlopCountAnalysis(model, dummy_input)
        return flops.total()
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        return None


def get_model_info(model: nn.Module, model_name: str) -> Dict:
    """Get comprehensive information about a model.

    Args:
        model: PyTorch model
        model_name: Name identifier for the model

    Returns:
        Dictionary with model information
    """
    params = count_parameters(model)
    flops = calculate_flops(model)

    return {
        "name": model_name,
        "params": params,
        "flops": flops,
        "params_m": round(params / 1e6, 2),  # In millions
        "flops_m": round(flops / 1e6, 2) if flops else None  # In millions
    }


def calculate_savings(
    routed_values: list,
    baseline_value: float,
    num_samples: int
) -> Dict:
    """Calculate resource savings compared to baseline.

    Args:
        routed_values: List of values (FLOPs or params) for each routed sample
        baseline_value: Value for always using the largest model
        num_samples: Total number of samples

    Returns:
        Dictionary with savings statistics
    """
    total_routed = sum(routed_values)
    total_baseline = baseline_value * num_samples
    savings = total_baseline - total_routed
    savings_percent = (savings / total_baseline * 100) if total_baseline > 0 else 0

    avg_routed = total_routed / num_samples if num_samples > 0 else 0

    return {
        "total_routed": int(total_routed),
        "total_baseline": int(total_baseline),
        "savings": int(savings),
        "savings_percent": round(savings_percent, 2),
        "avg_routed": int(avg_routed),
        "baseline_value": int(baseline_value)
    }
