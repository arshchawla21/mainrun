import math

import torch

from model.muon import Muon


class Optimizer:
    """Drives several optimizers as one (for the Muon + AdamW hybrid). Exposes the subset of the
    optimizer API the training loop + LR scheduler use."""
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()


class Scheduler:
    """Steps several LR schedulers together (one per optimizer in an Optimizer)."""
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self):
        for s in self.schedulers:
            s.step()

    def get_last_lr(self):
        return [lr for s in self.schedulers for lr in s.get_last_lr()]


def _wsd_lambda(args, max_steps: int):
    """LR multiplier for a warmup-stable-decay schedule, as a function of the scheduler's
    step index (0-indexed, called once per optimiser step):
      [0, warmup)        linear ramp 0 -> 1
      [warmup, decay0)   hold at 1 (the 'stable' plateau)
      [decay0, max]      decay 1 -> 0 with the chosen shape
    'sqrt' is the (1 - sqrt(progress)) decay from MiniCPM; 'cosine' is a half-cosine to 0."""
    warmup = max(1, int(args.warmup_frac * max_steps))
    decay  = max(1, int(args.decay_frac * max_steps))
    decay0 = max(warmup, max_steps - decay)        # step where decay begins

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        if step < decay0:
            return 1.0
        prog = min(1.0, (step - decay0) / max(1, max_steps - decay0))
        if args.decay_type == 'linear':
            return 1.0 - prog
        if args.decay_type == 'cosine':
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        if args.decay_type == 'sqrt':
            return 1.0 - math.sqrt(prog)
        raise ValueError(f"unknown decay_type {args.decay_type!r} (expected: linear | cosine | sqrt)")

    return lr_lambda


def _make_scheduler(opt, args, max_steps: int):
    """One scheduler for one optimiser, selected by args.optim_type. 'cosine' is the original
    CosineAnnealingLR (default -> unchanged behaviour); 'wsd' is warmup-stable-decay via LambdaLR."""
    if args.optim_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    if args.optim_type == 'wsd':
        return torch.optim.lr_scheduler.LambdaLR(opt, _wsd_lambda(args, max_steps))
    raise ValueError(f"unknown optim_type {args.optim_type!r} (expected: cosine | wsd)")


def build_optimizer(args, model, max_steps: int):
    """Return (optimizer, scheduler) for args.optim_alg. 'sgd'/'adamw' are single optimizers;
    'muonhybrid' is a hybrid: Muon on the 2D transformer weight matrices (lr=args.lr) + AdamW on
    everything else -- embeddings, tied head, LayerNorm gains, biases -- at args.lr_hybird.
    The LR schedule (cosine | wsd) is chosen by args.optim_type and applied to every group."""
    if args.optim_alg == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return opt, _make_scheduler(opt, args, max_steps)
    if args.optim_alg == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return opt, _make_scheduler(opt, args, max_steps)
    if args.optim_alg == 'muonhybrid':
        muon_p, adamw_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (muon_p if (p.ndim == 2 and 'emb' not in name) else adamw_p).append(p)
        muon = Muon(muon_p, lr=args.lr)                                   # 2D matrices: swept lr
        aux = torch.optim.AdamW(adamw_p, lr=args.lr_hybird,              # embeddings/head/norms/biases
                                weight_decay=args.weight_decay)
        opt = Optimizer([muon, aux])
        sched = Scheduler([_make_scheduler(muon, args, max_steps),
                           _make_scheduler(aux, args, max_steps)])
        return opt, sched
    raise ValueError(f"unknown optim_alg {args.optim_alg!r} (expected: sgd | adamw | muonhybrid)")
