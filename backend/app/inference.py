"""Model loading and inference logic."""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from config import (
    ROUTER_PATH, BRANCH_PATHS, DEVICE, NUM_CLASSES,
    CIFAR100_CLASSES
)
from app.utils import get_model_info, calculate_savings


class AdaptiveRouterInference:
    """Handles loading and inference for adaptive router system."""

    def __init__(self):
        """Initialize inference engine."""
        self.device = torch.device(DEVICE)
        self.router = None
        self.branches = []
        self.num_branches = 0
        self.models_info = {}
        self.class_names = CIFAR100_CLASSES

        print(f"Initializing inference engine on device: {self.device}")

    def load_models(
        self,
        router_path: Path,
        branch_paths: List[Path],
        router_model_class: Optional[nn.Module] = None,
        branch_model_class: Optional[nn.Module] = None
    ):
        """Load router and branch models.

        Args:
            router_path: Path to router model
            branch_paths: List of paths to branch models
            router_model_class: Optional router model class (if None, loads state dict only)
            branch_model_class: Optional branch model class (if None, loads state dict only)

        Note:
            If model classes are not provided, this method assumes the .pth files
            contain the full model (not just state_dict). For state_dict only,
            you need to provide the model architecture classes.
        """
        print("Loading models...")

        # Load router
        if router_path.exists():
            try:
                if router_model_class is not None:
                    self.router = router_model_class
                    self.router.load_state_dict(torch.load(router_path, map_location=self.device, weights_only=True))
                else:
                    # Try loading full model (trusted source)
                    self.router = torch.load(router_path, map_location=self.device, weights_only=False)

                self.router.to(self.device)
                self.router.eval()
                print(f"✓ Loaded router from {router_path}")

                # Get router info
                self.models_info['router'] = get_model_info(self.router, "Router")
            except Exception as e:
                raise RuntimeError(f"Failed to load router: {e}")
        else:
            raise FileNotFoundError(f"Router model not found at {router_path}")

        # Load branches
        self.branches = []
        self.models_info['branches'] = []

        for i, branch_path in enumerate(branch_paths):
            if branch_path.exists():
                try:
                    if branch_model_class is not None:
                        branch = branch_model_class
                        branch.load_state_dict(torch.load(branch_path, map_location=self.device, weights_only=True))
                    else:
                        # Try loading full model (trusted source)
                        branch = torch.load(branch_path, map_location=self.device, weights_only=False)

                    branch.to(self.device)
                    branch.eval()
                    self.branches.append(branch)

                    # Get branch info
                    branch_info = get_model_info(branch, f"Branch {i}")
                    branch_info['id'] = i
                    self.models_info['branches'].append(branch_info)

                    print(f"✓ Loaded branch {i} from {branch_path}")
                except Exception as e:
                    print(f"✗ Failed to load branch {i}: {e}")
            else:
                print(f"✗ Branch model not found at {branch_path}")

        self.num_branches = len(self.branches)

        if self.num_branches == 0:
            raise RuntimeError("No branch models loaded successfully")

        print(f"Successfully loaded {self.num_branches} branch models")

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        return_routing_logits: bool = False
    ) -> Tuple[List[Dict], Dict]:
        """Run inference on a batch of images.

        Args:
            images: Batch of preprocessed images [N, 3, 32, 32]
            return_routing_logits: Whether to return routing logits

        Returns:
            Tuple of (list of per-image results, aggregate metrics)
        """
        import time

        if self.router is None or len(self.branches) == 0:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        images = images.to(self.device)
        batch_size = images.shape[0]

        # Router inference with timing
        start_time = time.perf_counter()
        router_logits = self.router(images)  # [N, num_branches]
        router_time = time.perf_counter() - start_time

        router_probs = torch.softmax(router_logits, dim=1)
        routes = torch.argmax(router_logits, dim=1)  # [N]
        route_confidences = torch.max(router_probs, dim=1)[0]  # [N]

        # Branch inference with timing
        results = []
        routed_params = []
        routed_times = []

        # Get largest branch (Branch 4) for comparison
        largest_branch_idx = len(self.branches) - 1  # Last branch is largest
        largest_branch_model = self.branches[largest_branch_idx]

        for i in range(batch_size):
            route_idx = routes[i].item()
            route_confidence = route_confidences[i].item()

            # Get selected branch
            branch_model = self.branches[route_idx]
            branch_info = self.models_info['branches'][route_idx]

            # Run branch inference on single image with timing
            image = images[i:i+1]  # Keep batch dimension

            start_time = time.perf_counter()
            logits = branch_model(image)  # [1, num_classes]
            branch_time = time.perf_counter() - start_time

            probs = torch.softmax(logits, dim=1)

            pred_class = torch.argmax(logits, dim=1).item()
            confidence = torch.max(probs, dim=1)[0].item()
            pred_label = self.class_names[pred_class]

            # Run Branch 4 (largest) inference for comparison
            branch4_logits = largest_branch_model(image)
            branch4_probs = torch.softmax(branch4_logits, dim=1)
            branch4_pred_class = torch.argmax(branch4_logits, dim=1).item()
            branch4_confidence = torch.max(branch4_probs, dim=1)[0].item()
            branch4_pred_label = self.class_names[branch4_pred_class]

            result = {
                "predicted_class": pred_class,
                "predicted_label": pred_label,
                "confidence": round(confidence, 4),
                "route_chosen": route_idx,
                "route_confidence": round(route_confidence, 4),
                "branch4_predicted_class": branch4_pred_class,
                "branch4_predicted_label": branch4_pred_label,
                "branch4_confidence": round(branch4_confidence, 4)
            }

            if return_routing_logits:
                result["routing_logits"] = router_logits[i].cpu().tolist()

            results.append(result)

            # Track resources
            routed_params.append(branch_info['params'])
            # Total time = router time (amortized) + branch time
            routed_times.append((router_time / batch_size) + branch_time)

        # Calculate aggregate metrics
        metrics = self._calculate_metrics(
            results, routed_times, routed_params, batch_size
        )

        return results, metrics

    def _calculate_metrics(
        self,
        results: List[Dict],
        routed_times: List[float],
        routed_params: List[int],
        batch_size: int
    ) -> Dict:
        """Calculate aggregate metrics.

        Args:
            results: List of per-image results
            routed_times: List of inference times (seconds) for each routed sample
            routed_params: List of params for each routed sample
            batch_size: Total number of samples

        Returns:
            Dictionary of aggregate metrics
        """
        # Routing distribution
        routing_dist = {}
        for result in results:
            route = result['route_chosen']
            routing_dist[route] = routing_dist.get(route, 0) + 1

        # Average confidence
        avg_confidence = sum(r['confidence'] for r in results) / batch_size

        # Find largest model (baseline)
        largest_branch_idx = max(
            range(len(self.models_info['branches'])),
            key=lambda i: self.models_info['branches'][i]['params']
        )
        largest_branch = self.models_info['branches'][largest_branch_idx]

        # Calculate parameter savings
        params_savings = calculate_savings(
            routed_params,
            largest_branch['params'],
            batch_size
        )

        # Calculate time savings with proper benchmarking
        import time
        import torch

        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        largest_branch_model = self.branches[largest_branch_idx]

        # Warmup all models (3 runs)
        for _ in range(3):
            _ = self.router(dummy_input)
            for branch in self.branches:
                _ = branch(dummy_input)

        # Measure router time (average of 10 runs)
        router_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = self.router(dummy_input)
            router_times.append(time.perf_counter() - start)
        router_time_us = (sum(router_times) / len(router_times)) * 1_000_000

        # Measure each branch time (average of 10 runs)
        branch_times_us = []
        for branch in self.branches:
            times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = branch(dummy_input)
                times.append(time.perf_counter() - start)
            branch_times_us.append((sum(times) / len(times)) * 1_000_000)

        # Calculate routed time: router + average of actually used branches
        routed_time_components = []
        for result in results:
            route_idx = result['route_chosen']
            routed_time_components.append(router_time_us + branch_times_us[route_idx])

        # Baseline = router + largest branch
        baseline_time_us = router_time_us + branch_times_us[largest_branch_idx]

        time_savings = calculate_savings(
            routed_time_components,
            baseline_time_us,
            batch_size
        )

        # Calculate dominant labels per branch
        from collections import defaultdict, Counter
        branch_labels = defaultdict(list)
        for result in results:
            branch_idx = result['route_chosen']
            label = result['predicted_label']
            branch_labels[branch_idx].append(label)

        # Get top 5 most common labels per branch (if there are any)
        branch_label_distribution = {}
        for branch_idx in range(self.num_branches):
            if branch_idx in branch_labels and len(branch_labels[branch_idx]) > 0:
                label_counts = Counter(branch_labels[branch_idx])
                top_labels = [label for label, count in label_counts.most_common(5)]
                branch_label_distribution[f"Branch {branch_idx}"] = top_labels

        metrics = {
            "routing_distribution": routing_dist,
            "total_images": batch_size,
            "avg_confidence": round(avg_confidence, 4),
            "params_savings": params_savings,
            "time_savings": time_savings,
            "branch_label_distribution": branch_label_distribution if branch_label_distribution else None
        }

        return metrics

    def get_model_info_dict(self) -> Dict:
        """Get model architecture information.

        Returns:
            Dictionary with model info for all models
        """
        return {
            "num_branches": self.num_branches,
            "num_classes": NUM_CLASSES,
            "branches": self.models_info['branches'],
            "router": self.models_info['router']
        }
