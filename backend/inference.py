"""Routed inference: the router picks one branch per image and only that branch classifies it."""
import base64
import io
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .cifar100 import CLASSES, MEAN, STD
from .models import load_models

WEIGHTS_DIR = Path(__file__).parent / "weights"
INPUT_SIZE = 32  # CIFAR-100 resolution; every upload is resized to this
BATCH_SIZE = 64  # bounds memory when thousands of images arrive in one request


def load_image(file):
    """Open an uploaded file and shrink it to the 32x32 RGB image the models see."""
    image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)


def data_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def top1(logits):
    """Per row: the winning index and its softmax probability."""
    return logits.argmax(dim=1).tolist(), logits.softmax(dim=1).amax(dim=1).tolist()


@torch.inference_mode()
def latency_ms(model, runs=20, warmup=3):
    """Median time of a single-image forward pass on this machine."""
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    times = []
    for _ in range(warmup + runs):
        start = time.perf_counter()
        model(x)
        times.append(time.perf_counter() - start)
    return 1000 * statistics.median(times[warmup:])


def describe(name, model):
    params = sum(p.numel() for p in model.parameters())
    return {"name": name, "params": params, "latency_ms": round(latency_ms(model), 3)}


def savings(routed, baseline):
    percent = 100 * (1 - routed / baseline)
    return {"routed": round(routed, 3), "baseline": round(baseline, 3), "savings_percent": round(percent, 2)}


class RoutedClassifier:
    """The router and its branches, loaded once and measured at startup."""

    def __init__(self, weights_dir=WEIGHTS_DIR):
        self.router, self.branches = load_models(weights_dir)
        self.mean, self.std = torch.tensor(MEAN), torch.tensor(STD)

        branches = [describe(f"Branch {i}", branch) for i, branch in enumerate(self.branches)]
        # Everything is compared against always using the largest branch.
        self.baseline = max(range(len(branches)), key=lambda i: branches[i]["params"])
        self.info = {
            "router": describe("Router", self.router),
            "branches": branches,
            "baseline_branch": self.baseline,
        }

    @torch.inference_mode()
    def predict(self, images):
        """Classify 32x32 RGB PIL images. Returns (one result per image, batch metrics)."""
        pixels = torch.from_numpy(np.stack([np.asarray(image) for image in images]))  # N x 32 x 32 x 3, uint8
        rows = []
        for chunk in pixels.split(BATCH_SIZE):
            x = ((chunk.float() / 255 - self.mean) / self.std).permute(0, 3, 1, 2)
            route_logits = self.router(x)
            routes = route_logits.argmax(dim=1)

            # The largest branch sees every image, only so the UI can show what it would have said.
            baseline_logits = self.branches[self.baseline](x)
            logits = baseline_logits.clone()
            for i, branch in enumerate(self.branches):
                routed_here = routes == i
                if i != self.baseline and routed_here.any():
                    logits[routed_here] = branch(x[routed_here])

            rows += zip(*top1(logits), *top1(route_logits), *top1(baseline_logits))

        results = [
            {
                "image_preview": data_uri(image),
                "predicted_class": cls,
                "predicted_label": CLASSES[cls],
                "confidence": round(confidence, 4),
                "route_chosen": route,
                "route_confidence": round(route_confidence, 4),
                "baseline_class": baseline_cls,
                "baseline_label": CLASSES[baseline_cls],
                "baseline_confidence": round(baseline_confidence, 4),
            }
            for image, (cls, confidence, route, route_confidence, baseline_cls, baseline_confidence)
            in zip(images, rows)
        ]
        return results, self.summarize(results)

    def summarize(self, results):
        """Batch metrics: where the images went, and the cost of that versus the largest branch."""
        n = len(results)
        branches = self.info["branches"]
        largest = branches[self.baseline]
        router_ms = self.info["router"]["latency_ms"]

        labels = [Counter() for _ in branches]  # predicted labels, per branch that produced them
        for result in results:
            labels[result["route_chosen"]][result["predicted_label"]] += 1
        counts = [sum(seen.values()) for seen in labels]

        def average(key):  # mean over images of the chosen branch's `key`
            return sum(count * branch[key] for count, branch in zip(counts, branches)) / n

        return {
            "total_images": n,
            "avg_confidence": round(sum(result["confidence"] for result in results) / n, 4),
            "routing": [
                {"count": count, "top_labels": [label for label, _ in seen.most_common(5)]}
                for count, seen in zip(counts, labels)
            ],
            # Size of the branch that ran; the router's own parameters are not counted.
            "params_savings": savings(average("params"), largest["params"]),
            # Per-image latency with the router pass on both sides:
            # router + chosen branch, versus router + largest branch.
            "time_savings": savings(router_ms + average("latency_ms"), router_ms + largest["latency_ms"]),
        }
