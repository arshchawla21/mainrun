"""
Hybrid Muon -- MomentUm Orthogonalized by Newton-schulz (Keller Jordan et al., 2024).

Muon updates 2D hidden weight matrices: it takes the momentum buffer and orthogonalises 
it with a quintic Newton-Schulz iteration before the step. It must NOT be used for 
embeddings, the (tied) LM head, LayerNorm gains, or biases -- those stay
on AdamW. See build_optimizer() in train.py for the hybrid wiring.
"""
import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalize G (2D) via a quintic Newton-Schulz iteration. Runs in bf16 on
    CUDA (the iteration is robust to the low precision) and fp32 on CPU."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(dtype)
    X = X / (X.norm() + eps)                 # normalize so the iteration's spectral radius is bounded
    transposed = G.size(0) > G.size(1)
    if transposed:                           # iterate on the smaller dimension
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2D matrices only. Pair with AdamW for all non-matrix / embedding params.
    lr is much larger than AdamW's (typically ~0.02-0.05)."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mom = group["momentum"]
            wd = group.get("weight_decay", 0.0)        # decoupled WD (0 = off -> exact original behaviour)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                if wd > 0:                             # decoupled weight decay, scaled by the current LR
                    p.mul_(1.0 - group["lr"] * wd)
                # shape-aware scale -> update RMS stays ~consistent across matrix aspect ratios
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-group["lr"] * scale)
