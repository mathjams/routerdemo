import os
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from dataclasses import dataclass
from typing import Deque, List


def reset_module(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()


def save_checkpoint(network, branch_optimizers, router_optimizer, step, path):
    ckpt = {
        "step": step,
        "router_state": network.router.state_dict(),
        "branch_states": [b.state_dict() for b in network.models],
        "router_opt_state": router_optimizer.state_dict() if router_optimizer is not None else None,
        "branch_opt_states": [opt.state_dict() for opt in branch_optimizers] if branch_optimizers is not None else None,
        "costs": network.costs.detach().cpu(),
        "branches": network.branches,
    }
    torch.save(ckpt, path)
    print("Saved:", path)


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_imagenet_dataloaders(data_dir, batch_size=64, num_workers=2, subset_size=None, download=True):
    normalize = transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    )
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
