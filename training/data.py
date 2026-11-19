"""CIFAR-100 loading and the small batching helpers the training loop uses."""
import random
from dataclasses import dataclass
from typing import Deque, List

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from backend.cifar100 import MEAN, STD


def get_cifar100_dataloaders(data_dir, batch_size=64, num_workers=2, subset_size=None, download=True):
    """Train and test loaders. `subset_size` trains on a random subset, for quick experiments."""
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
    )
    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = torchvision.datasets.CIFAR100(data_dir, train=True, download=download, transform=train_transforms)
    val_dataset = torchvision.datasets.CIFAR100(data_dir, train=False, download=download, transform=val_transforms)

    if subset_size is not None:
        train_dataset = Subset(train_dataset, random.sample(range(len(train_dataset)), min(subset_size, len(train_dataset))))
        val_dataset = Subset(val_dataset, random.sample(range(len(val_dataset)), min(max(1, subset_size // 10), len(val_dataset))))

    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    if num_workers > 0:
        common["prefetch_factor"] = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def infinite_loader(loader: DataLoader):
    while True:
        yield from loader


class CUDAPrefetcher:
    """Endless batches, each already copied to the GPU while the previous one was being used."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.it = iter(loader)
        self.next_x = None
        self.next_y = None
        self._preload()

    def _preload(self):
        try:
            x, y = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            x, y = next(self.it)

        with torch.cuda.stream(self.stream):
            self.next_x = x.to(self.device, non_blocking=True)
            self.next_y = y.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        x, y = self.next_x, self.next_y
        self._preload()
        return x, y


@dataclass
class TensorQueue:
    """FIFO of tensor rows: add chunks of any size, pop exact-size batches."""

    chunks: Deque[torch.Tensor]
    total: int = 0

    def add(self, t: torch.Tensor) -> None:
        if t.numel() == 0:
            return
        self.chunks.append(t)
        self.total += int(t.size(0))

    def pop(self, n: int) -> torch.Tensor:
        assert self.total >= n, "Not enough data in buffer"
        parts: List[torch.Tensor] = []
        need = n
        while need > 0:
            t = self.chunks[0]
            if int(t.size(0)) <= need:
                parts.append(self.chunks.popleft())
                need -= int(t.size(0))
            else:
                parts.append(t[:need])
                self.chunks[0] = t[need:]
                need = 0
        self.total -= n
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
