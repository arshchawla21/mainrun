import utils
import math, random, time
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from datasets import load_dataset
from tqdm import tqdm
import structlog
import shutil
import subprocess

from model.tokenizer import train_tokenizer, Tokenizer
from model.gpt import GPTConfig, CausalSelfAttention, GPT
from model.optimizer import build_optimizer

@dataclass
class Hyperparameters:
    # phase A
    block_size: int = 128
    batch_size: int = 64
    vocab_size: int = 16_000
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.1
    lr: float = 6e-3
    lr_hybird: float = 3e-4
    weight_decay: float = 0.01
    evals_per_epoch: int = 3

    # improvements
    token_type: str = 'bpe'   # 'bpe' | 'unigram' | 'wordpiece' | 'superbpe' | 'wordlevel'
    transition_ratio: float = 0.75  # superbpe only: fraction of vocab spent on stage-1 subwords
    optim_type: str = 'cosine' # LR schedule: 'cosine' (CosineAnnealingLR) | 'wsd' (warmup-stable-decay)
    optim_alg: str = 'sgd'   # gradient-descent algorithm: 'sgd' | 'adamw' | 'muonhybrid'
    warmup_frac: float = 0.05  # wsd only: fraction of steps spent in linear warmup (inert for cosine)
    decay_frac: float = 0.2    # wsd only: fraction of steps spent in the final decay (inert for cosine)
    decay_type: str = 'cosine' # wsd only: decay shape -- 'linear' | 'cosine' | 'sqrt' (1-sqrt)
    norm_type: str = 'layernorm' # 'layernorm' | 'rmsnorm'
    mlp_type: str = 'gelu' # 'gelu' | 'swiglu'
    pos_type : str = 'learned' # 'learned' | 'rope'
    qk_norm: bool = False
    bias: str = 'default'   # 'default' = original (Linear + LayerNorm biases) | 'off' = no bias anywhere
    title_masking: bool = False 

    # phase B
    gpt_v2: bool = False # if true, we use the new model (gpt_v2, with our improvements from A present)
    kv_cache: str = False 
    attn_type: str = 'mha'   # v2 only: mha | single_head | value_residual | output_gated | differential
    residual: str = 'none'   # v2 only: none | unet | embedding_shortcut | layerscale
    attn_temp: bool = False  # v2 only: per-(layer,head) learned attention temperature
    label_smoothing: float = 0.0  # v2 only: soften targets in the TRAINING loss (eval stays hard CE)
    rdrop: float = 0.0       # R-Drop: weight on the symmetric-KL between two dropout passes (0 = off)
    amp: bool = False        # bf16 autocast on the training forward (fits bigger models + R-Drop on 12GB)

    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"

    # experiment tracking
    wandb: bool = False
    wandb_project: str = "mainrun"
    wandb_run_name: Optional[str] = None

    # checkpointing
    save_checkpoint: bool = False
    out_root: str = "runs"

def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = open(log_file, 'w')

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()

        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()

            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)

    return DualLogger(file_handler)

logger = None

def _git_short_hash() -> str:
    """Short hash of the current commit, so a run folder is traceable to the code that made it."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "nogit"

def get_titles(num_titles: int, seed: int, val_frac: float) -> str:
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]

def get_batch(split_ids: torch.Tensor, ptr: int, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

def eval_train_val(model, train_ids, train_text, val_ids, val_text,
                   block_size, batch_size, device):
    """
    Standalone twin of evaluate(): summed token NLL / characters of `text` (nats per char).
    Returns (train_loss, val_loss) so both numbers come from one identical code path.
    Does NOT touch evaluate() or mainrun.log -- used only to populate loss.log for plotting.
    """
    def _split(ids, text):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(ids, block_size, batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                total += F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum').item()
        model.train()
        return total / len(text)
    return _split(train_ids, train_text), _split(val_ids, val_text)

def save_run(result: dict, base_dir: str = "../sweeps", tag: str = "run") -> Path:
    """
    Persist one finished run into  base_dir/{tag}_{time}_valloss{X}/  containing:
      - config.json    : the exact Hyperparameters used
      - model.pt       : weights + GPTConfig + final val loss (everything needed to sample)
      - tokenizer.json : the trained tokenizer (so samples decode correctly)
      - mainrun.log    : a copy of the training log
      - loss.log       : a copy of the train/val per-char loss log (for plotting)
    """
    args     = result["args"]
    val_loss = result["val_loss"]

    run_dir  = Path(base_dir) / f"{tag}_valloss{val_loss:.4f}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. which hyperparams were used
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 2. checkpoint for sampling later
    torch.save({
        "model_state_dict": result["model"].state_dict(),
        "config":           result["cfg_dict"],   # rebuild with GPT(GPTConfig(**config))
        "args":             vars(args),
        "val_loss":         val_loss,
        "step":             result["step"],
    }, run_dir / "model.pt")

    # 3. tokenizer (raw tokenizers.Tokenizer supports .save)
    try:
        result["raw_tok"].save(str(run_dir / "tokenizer.json"))
    except Exception as e:
        print(f"warning: could not save tokenizer ({e})")

    # 4. copy the logs in (lowkey just cp them)
    try:
        shutil.copy(args.log_file, run_dir / "mainrun.log")
    except Exception as e:
        print(f"warning: could not copy log ({e})")
    try:
        loss_log = Path(args.log_file).with_name("loss.log")
        if loss_log.exists():
            shutil.copy(loss_log, run_dir / "loss.log")
    except Exception as e:
        print(f"warning: could not copy loss.log ({e})")

    print(f"saved run -> {run_dir}")
    return run_dir

def main(args: Optional[Hyperparameters] = None) -> dict:
    if args is None:
        args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    global logger
    logger = configure_logging(args.log_file)

    # separate loss log, lives next to mainrun.log (e.g. ./logs/loss.log)
    loss_log_path = Path(args.log_file).with_name("loss.log")
    loss_log_path.parent.mkdir(parents=True, exist_ok=True)
    loss_fh = open(loss_log_path, "w")

    wb = None
    if args.wandb:
        import wandb as wb
        wb.init(project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args))

    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)

    eos_token = "<eos>"
    raw_tok = train_tokenizer(args.token_type, train_titles+val_titles, args.vocab_size,
                              eos_token=eos_token, transition_ratio=args.transition_ratio)
    tok = Tokenizer(raw_tok)
    eos_id = raw_tok.token_to_id(eos_token)   # for title_masking: marks the <eos> title boundaries
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

    # fixed slice of train, scored the same way as val (same size -> same cost & comparable).
    # swap for (train_ids, train_text) if you want train loss over the whole train set.
    train_eval_titles = train_titles[:len(val_titles)]
    train_eval_text = eos_token.join(train_eval_titles) + eos_token
    train_eval_ids = torch.tensor(tok.encode(train_eval_text), dtype=torch.long)

    batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    cfg_dict = dict(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
        norm_type  = args.norm_type,
        mlp_type   = args.mlp_type,
        pos_type   = args.pos_type,
        qk_norm    = args.qk_norm,
        bias       = args.bias,
        title_masking = args.title_masking,
        eos_id     = eos_id,
        attn_type  = args.attn_type,
        residual   = args.residual,
        attn_temp  = args.attn_temp,
        label_smoothing = args.label_smoothing,
    )
    cfg = GPTConfig(**cfg_dict)

    if args.gpt_v2:
        from model.gpt_v2 import GPT2
        model = GPT2(cfg).to(device)       # locked Phase A backbone; arch flags above are ignored
    else:
        model = GPT(cfg).to(device)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)

    opt, scheduler = build_optimizer(args, model, max_steps)

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    ptr = 0
    step = 0
    t0 = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
                step += 1
                xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.batch_size, device)
                amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (args.amp and device == "cuda") else nullcontext()
                with amp_ctx:
                    if args.rdrop > 0:               # R-Drop: two dropout passes + symmetric-KL consistency
                        lg1, l1 = model(xb, yb)
                        lg2, l2 = model(xb, yb)
                        lp1, lp2 = F.log_softmax(lg1, -1), F.log_softmax(lg2, -1)
                        kl = 0.5 * ((lp1.exp() * (lp1 - lp2)).sum(-1) +
                                    (lp2.exp() * (lp2 - lp1)).sum(-1)).mean()
                        loss = 0.5 * (l1 + l2) + args.rdrop * kl
                    else:
                        _, loss = model(xb, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()

                elapsed = time.time() - t0
                logger.log("training_step",
                        step=step,
                        max_steps=max_steps,
                        loss=loss.item(),
                        elapsed_time=elapsed,
                        prnt=False)

                if wb is not None:
                    wb.log({"train_loss": loss.item(),
                            "lr": scheduler.get_last_lr()[0]}, step=step)

                if step == 1 or step % eval_interval == 0 or step == max_steps:
                    val_loss = evaluate()
                    logger.log("validation_step",
                            step=step,
                            max_steps=max_steps,
                            loss=val_loss,
                            elapsed_time=elapsed)
                    if wb is not None:
                        wb.log({"val_loss": val_loss}, step=step)

                    # --- separate train/val measurement -> loss.log (for plotting) ---
                    tr, vl = eval_train_val(model,
                                            train_eval_ids, train_eval_text,
                                            val_ids, val_text,
                                            args.block_size, args.batch_size, device)
                    loss_fh.write(json.dumps({
                        "step":  step,
                        "epoch": round(step / batches, 4),
                        "train": tr,
                        "val":   vl,
                    }) + "\n")
                    loss_fh.flush()
    finally:
        if wb is not None:
            wb.finish()
        try:
            logger.file_handler.close()
        except Exception:
            pass
        try:
            loss_fh.close()
        except Exception:
            pass

    last_val_loss = evaluate()

    return {
        "args":     args,
        "model":    model,
        "tok":      tok,
        "raw_tok":  raw_tok,
        "cfg_dict": cfg_dict,
        "val_loss": last_val_loss,
        "step":     step,
    }

if __name__ == "__main__":
    result = main()
    if result["args"].save_checkpoint:
        save_run(result, base_dir=result["args"].out_root, tag="train")