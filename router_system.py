import torch
import torch.nn as nn


# NOTE training router does not train the branches
# set requires_grad=False on all branches before training
class RouterSystem(nn.Module):
    def __init__(
        self,
        router: nn.Module,
        branches: list[nn.Module],
        costs: torch.Tensor,
    ):
        super().__init__()
        self.router = router
        self.models = nn.ModuleList(branches)
        self.branches = len(branches)
        assert costs.shape == (self.branches,)
        self.register_buffer("costs", costs)
    @torch.no_grad()
    def router_instaloss(
        self,
        x: torch.Tensor,        # (B, ...)
        target: torch.Tensor,   # (B,)
        cost_weight: float = 1.0,
        t_router: float = 1.0,
        beta: float = 15,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes Option-A instantaneous router loss PER SAMPLE (no grad):
          teacher q = softmax(-beta * branch_losses)
          router  p = softmax(router_logits / t_router)
          instaloss = CE(q, p) = -sum_i q_i log p_i

        Returns:
          router_loss_per_sample: (B,)
          branch_losses: (B, branches)  (includes cost_weight * costs)
          routing_logits: (B, branches)
        """
        # Get device of router (main device)
        router_device = next(self.router.parameters()).device

        # Handle multi-GPU: move data to each branch's device
        branch_logits_list = []
        for m in self.models:
            model_device = next(m.parameters()).device
            x_i = x.to(model_device) if model_device != x.device else x
            logits = m(x_i)
            # Move back to router device for stacking
            logits = logits.to(router_device) if logits.device != router_device else logits
            branch_logits_list.append(logits)

        branch_logits = torch.stack(branch_logits_list, dim=0)  # (branches, B, C)

        # Move target to router device as well
        target_device = target.to(router_device) if target.device != router_device else target

        # CE per-branch per-sample -> (branches, B) -> (B, branches)
        branch_losses = nn.functional.cross_entropy(
            branch_logits.movedim(-1, 1),  # (branches, C, B)
            target_device.unsqueeze(0).expand(self.branches, *target_device.shape),
            reduction="none",
        ).movedim(0, -1) + cost_weight * self.costs  # (B, branches)

        # Ensure x is on router device
        x_router = x.to(router_device) if x.device != router_device else x
        routing_logits = self.router(x_router)  # (B, branches)

        p = nn.functional.softmax(routing_logits / max(t_router, 1e-8), dim=-1)         # (B, branches)
        q = nn.functional.softmax(-beta * branch_losses, dim=-1)                        # (B, branches)

        router_loss_per_sample = -(q * torch.log(p.clamp_min(1e-9))).sum(dim=-1)  # (B,)

        return router_loss_per_sample, branch_losses, routing_logits
    def train_step(
        self,
        x: torch.Tensor,        # (B, ...)
        target: torch.Tensor,   # (B,)
        optimizer: torch.optim.Optimizer,
        cost_weight: 1.0,
        t_router = 1.0, 
        beta = 3.0, 
        ls = 0.0, 
        lb_weight = 1.5,
    ) -> dict:

        self.router.train()
        optimizer.zero_grad(set_to_none=True)

        # Get device of router (main device)
        router_device = next(self.router.parameters()).device

        with torch.no_grad():
            # branch_losses: (B, branches)
            # Handle multi-GPU: move data to each branch's device
            branch_logits_list = []
            for m in self.models:
                model_device = next(m.parameters()).device
                x_i = x.to(model_device, non_blocking=True) if model_device != x.device else x
                logits = m(x_i)
                # Move back to router device for stacking
                logits = logits.to(router_device, non_blocking=True) if logits.device != router_device else logits
                branch_logits_list.append(logits)

            branch_logits = torch.stack(branch_logits_list, dim=0)  # (branches, B, C)

            # Move target to router device as well
            target_device = target.to(router_device, non_blocking=True) if target.device != router_device else target

            branch_losses = nn.functional.cross_entropy(
                branch_logits.movedim(-1, 1),  # (branches, C, B)
                target_device.unsqueeze(0).expand(self.branches, *target_device.shape),
                reduction="none",
            ).movedim(0, -1) + cost_weight*self.costs  # (B, branches)
            optimal_indices = torch.argmin(branch_losses, dim=-1)

        # Ensure x is on router device
        x_router = x.to(router_device, non_blocking=True) if x.device != router_device else x
        routing_logits = self.router(x_router)

        # Teacher distribution (from branch losses)
        teacher_probs = torch.softmax(-beta * branch_losses, dim=-1)
        teacher_probs = (1 - ls) * teacher_probs + ls / self.branches  # label smoothing

        # Router distribution
        routing_log_probs = torch.log_softmax(routing_logits / t_router, dim=-1)
        router_probs = routing_log_probs.exp()
        distill_loss = -(teacher_probs * routing_log_probs).sum(dim=-1).mean()
				
        E = self.branches
        B = router_probs.size(0)
				
				# Switch-style load balancing: encourages ~uniform expert usage even with confident routing
        importance = router_probs.sum(dim=0) / B                         # (E,)
        top1 = router_probs.argmax(dim=-1)                               # (B,)
        load = torch.bincount(top1, minlength=E).float() / B             # (E,)
        balance_loss = E * torch.sum(importance * load)                  # scalar, min near uniform
				
				# Per-example entropy (minimize => confident / near one-hot routing)
        #ent_loss = -(router_probs * routing_log_probs).sum(dim=-1).mean()
				
        router_loss = distill_loss + lb_weight * balance_loss #+ ent_weight * ent_loss
				
        router_loss.backward()
        optimizer.step()
        with torch.no_grad():
            predicted_indices = torch.argmax(routing_logits, dim=-1)
            accuracy = (predicted_indices == optimal_indices).float().mean()

        return {
            "router_loss": router_loss.detach().item(),
            "distill_loss": distill_loss.detach().item(),
            "balance_loss": balance_loss.detach().item(),
            "router_acc": accuracy.detach().item(),
            "cost_weight": float(cost_weight),
            "t_router": float(t_router),
            "beta": float(beta),
            "ls": float(ls),
            "accuracy": float(accuracy),
        }

    @torch.no_grad()
    def route(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        epsilon: float = 0.0
    ) -> torch.Tensor:
        self.router.eval()
        logits = self.router(x)

        if temperature == 0.0:
            return torch.argmax(logits, dim=-1)

        probs = torch.softmax(logits / temperature, dim=-1)
        if epsilon > 0:
            probs = probs + epsilon
            probs = probs / probs.sum(dim=-1, keepdim=True)

        return torch.multinomial(probs.view(-1, self.branches), num_samples=1).view(*probs.shape[:-1])

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        temperature: float = 0.0
    ) -> tuple[torch.Tensor, int]:
        self.router.eval()
        logits = self.router(x)

        if temperature == 0.0:
            route = torch.argmax(logits).item()
        else:
            route = torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1).item()

        return self.models[route](x), route


class Padded(nn.Module):
    def __init__(self, context: int, model: nn.Module):
        super().__init__()
        self.d_in = context
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) > self.d_in:
            x = x[..., -self.d_in:]
        elif x.size(-1) < self.d_in:
            x = torch.cat(
                [torch.zeros(*x.shape[:-1], self.d_in - x.size(-1), device=x.device, dtype=x.dtype), x],
                dim=-1,
            )
        return self.model(x)
