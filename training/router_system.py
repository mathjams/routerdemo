"""The router together with its branches, and the losses used to train the router."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterSystem(nn.Module):
    """A router plus N branches. Each branch has a cost; cheaper branches are preferred.

    Branches may live on different GPUs; everything is gathered on the router's device.
    Training the router does not train the branches (freeze them first).
    """

    def __init__(self, router: nn.Module, branches: list, costs: torch.Tensor):
        super().__init__()
        self.router = router
        self.branches = nn.ModuleList(branches)
        self.num_branches = len(branches)
        assert costs.shape == (self.num_branches,)
        self.register_buffer("costs", costs)

    @property
    def device(self):
        return next(self.router.parameters()).device

    @torch.no_grad()
    def branch_losses(self, x, target, cost_weight):
        """Cross-entropy of every branch on every sample, plus its cost -> (B, num_branches)."""
        logits = torch.stack([
            branch(x.to(next(branch.parameters()).device, non_blocking=True)).to(self.device, non_blocking=True)
            for branch in self.branches
        ])  # (num_branches, B, C)
        target = target.to(self.device, non_blocking=True)
        losses = F.cross_entropy(
            logits.movedim(-1, 1),  # (num_branches, C, B)
            target.unsqueeze(0).expand(self.num_branches, *target.shape),
            reduction="none",
        )
        return losses.movedim(0, -1) + cost_weight * self.costs

    @torch.no_grad()
    def router_instaloss(self, x, target, cost_weight=1.0, t_router=1.0, beta=15):
        """Per-sample router loss, without gradients -> (B,).

        The teacher is q = softmax(-beta * branch_losses) and the router is
        p = softmax(router_logits / t_router); the loss is the cross-entropy -sum_i q_i log p_i.
        """
        branch_losses = self.branch_losses(x, target, cost_weight)
        routing_logits = self.router(x.to(self.device))
        p = F.softmax(routing_logits / max(t_router, 1e-8), dim=-1)
        q = F.softmax(-beta * branch_losses, dim=-1)
        return -(q * torch.log(p.clamp_min(1e-9))).sum(dim=-1)

    def train_step(self, x, target, optimizer, cost_weight, t_router=1.0, beta=3.0, ls=0.0, lb_weight=1.5):
        """One optimiser step on the router: imitate the teacher, and keep every branch in use."""
        self.router.train()
        optimizer.zero_grad(set_to_none=True)

        branch_losses = self.branch_losses(x, target, cost_weight)
        optimal_indices = torch.argmin(branch_losses, dim=-1)
        routing_logits = self.router(x.to(self.device, non_blocking=True))

        # Teacher distribution from the branch losses, with optional label smoothing.
        teacher_probs = torch.softmax(-beta * branch_losses, dim=-1)
        teacher_probs = (1 - ls) * teacher_probs + ls / self.num_branches

        routing_log_probs = torch.log_softmax(routing_logits / t_router, dim=-1)
        router_probs = routing_log_probs.exp()
        distill_loss = -(teacher_probs * routing_log_probs).sum(dim=-1).mean()

        # Switch-style load balancing: encourages ~uniform branch usage even with confident routing.
        batch = router_probs.size(0)
        importance = router_probs.sum(dim=0) / batch
        load = torch.bincount(router_probs.argmax(dim=-1), minlength=self.num_branches).float() / batch
        balance_loss = self.num_branches * torch.sum(importance * load)  # smallest near uniform

        router_loss = distill_loss + lb_weight * balance_loss
        router_loss.backward()
        optimizer.step()

        accuracy = (routing_logits.argmax(dim=-1) == optimal_indices).float().mean()
        return {
            "router_loss": router_loss.item(),
            "distill_loss": distill_loss.item(),
            "balance_loss": balance_loss.item(),
            "accuracy": accuracy.item(),  # how often the router picks the lowest-loss branch
        }

    @torch.no_grad()
    def route(self, x, temperature=1.0, epsilon=0.0):
        """Pick a branch per sample: argmax at temperature 0, otherwise sampled (epsilon adds exploration)."""
        self.router.eval()
        logits = self.router(x)
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1)

        probs = torch.softmax(logits / temperature, dim=-1)
        if epsilon > 0:
            probs = probs + epsilon
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs.view(-1, self.num_branches), num_samples=1).view(*probs.shape[:-1])
