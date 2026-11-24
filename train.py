# Trains build_model() from model.py and writes weights.pt.
#   Submission run (train.npz + val.npz):   python train.py
#   Honest evaluation (train.npz only):     python train.py --holdout     (reports accuracy on val.npz)
#
# The model is small (d = 40), so a training step is limited by the number of GPU kernel launches and memory
# passes, not by arithmetic. This script therefore makes each step cheaper, so more epochs fit in the time limit:
#   * torch.compile fuses the many small element-wise ops into a few kernels (training only; model.py and the
#     saved weights are plain PyTorch). If compilation fails, training falls back to eager mode automatically;
#   * fused AdamW (one kernel for all parameters) and a foreach EMA update (one kernel for all EMA weights);
#   * no CPU-GPU synchronisation inside the loop (the loss is read only when a log line is printed).
# Training is time-budgeted: the cosine schedule follows elapsed training time (compilation included, evaluation
# excluded), so the run ends inside --minutes on whatever T4 runs it.
# Optional length curriculum: for the first --crop-frac of the time, train on random crops of --crop-len slots
# (cheaper steps, so more optimiser updates), then on full 1024-slot sequences.
import argparse
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build_model

T0 = 32                     # targets x[:, 32:] are scored
N_FINAL = 9000              # sequences in train + val; progress is reported in passes over 9,000 sequences
DEFAULT_COMPILE = "default"    # @@COMPILE@@  one of: none, default, reduce-overhead (set by the notebook benchmark)


def show(msg):
    """Lines for the person start with '» '; the notebook prints only these (plus RESULT/BENCH/DONE summaries)."""
    print("» " + msg, flush=True)


@torch.no_grad()
def evaluate(model, x, device, bs=32):
    model.eval()
    hit = nll = n = 0.0
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device)
        lg = model(xb)[:, T0 - 1:-1].float()
        y = xb[:, T0:]
        hit += (lg.argmax(-1) == y).sum().item()
        nll += F.cross_entropy(lg.reshape(-1, 64), y.reshape(-1), reduction="sum").item()
        n += y.numel()
    return hit / n, nll / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--holdout", action="store_true", help="train on train.npz only and report val.npz")
    ap.add_argument("--minutes", type=float, default=18.0, help="training-time budget, compilation included")
    ap.add_argument("--compile", default=DEFAULT_COMPILE, choices=["none", "default", "reduce-overhead"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.009)
    ap.add_argument("--wd", type=float, default=0.15)
    ap.add_argument("--warmup-steps", type=int, default=200, help="linear warm-up, counted in optimiser steps")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--crop-len", type=int, default=0, help="length of the random crops of the first phase (0 = off)")
    ap.add_argument("--crop-frac", type=float, default=0.0, help="fraction of the time budget spent on crops")
    ap.add_argument("--eval-every", type=float, default=0, help="with --holdout: minutes between quick evals")
    ap.add_argument("--log-every", type=float, default=60, help="seconds between progress rows")
    ap.add_argument("--bench", type=int, default=0, help="only time this many steps, then exit (no saving)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="weights.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda = device == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True   # fixed shapes: let cuDNN pick its fastest GRU / conv kernels

    # ---- data (loading is excluded from the time limit) ----
    xtr = torch.from_numpy(np.load(f"{a.data_dir}/train.npz")["x"].astype(np.int64))
    xva = torch.from_numpy(np.load(f"{a.data_dir}/val.npz")["x"].astype(np.int64))
    if not a.holdout:
        xtr = torch.cat((xtr, xva))
    xtr = xtr.to(device)
    n_seq, seq_len = xtr.shape
    crop_on = 0 < a.crop_len < seq_len and a.crop_frac > 0

    # ---- model, EMA copy, optimiser ----
    model = build_model().to(device)
    n_par = sum(v.numel() for v in model.state_dict().values())
    print(f"[setup] parameters {n_par:,} | sequences {n_seq} | device {device} | budget {a.minutes} min | "
          f"compile {a.compile} | lr {a.lr} wd {a.wd} bs {a.bs} | crops {a.crop_len if crop_on else 'off'}"
          f"{' for ' + str(a.crop_frac) + ' of the time' if crop_on else ''}", flush=True)
    assert n_par <= 50_000, "parameter budget exceeded"
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    for m in list(model.modules()) + list(ema.modules()):
        if isinstance(m, nn.RNNBase):
            m.flatten_parameters()
    model_params = list(model.parameters())
    ema_params = list(ema.parameters())

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    other = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    groups = [{"params": decay, "weight_decay": a.wd}, {"params": other, "weight_decay": 0.0}]
    try:
        opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.98), eps=1e-8, fused=cuda)
    except (RuntimeError, TypeError) as e:
        print(f"[setup] fused AdamW unavailable ({e}); using the default implementation", flush=True)
        opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.98), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    train_model = model
    if a.compile != "none":
        try:
            train_model = torch.compile(model, mode=a.compile)
        except Exception as e:
            show(f"⚠️  speed-up (torch.compile) unavailable here, training in normal mode ({type(e).__name__})")
            train_model = model

    def lr_at(p, step):  # step-based linear warm-up x time-based cosine decay to 0
        return a.lr * min(1.0, (step + 1) / a.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    def forward_loss(net, xb):
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            lg = net(xb[:, :-1])
        return F.cross_entropy(lg.float().reshape(-1, 64), xb[:, 1:].reshape(-1))

    def train_step(xb, lr, dcy):
        nonlocal train_model
        for pg in opt.param_groups:
            pg["lr"] = lr
        if train_model is not model:
            try:
                if a.compile == "reduce-overhead":
                    torch.compiler.cudagraph_mark_step_begin()
                loss = forward_loss(train_model, xb)
            except Exception as e:  # compilation problem (first step or a new input length): continue uncompiled
                show(f"⚠️  speed-up (torch.compile) failed, training in normal mode ({type(e).__name__})")
                print(f"[warn ] compile error: {str(e)[:300]}", flush=True)
                train_model = model
                loss = forward_loss(model, xb)
        else:
            loss = forward_loss(model, xb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model_params, 1.0)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, model_params, 1.0 - dcy)
        return loss.detach()

    # ---- batches: shuffled passes over the data, without replacement, built on the GPU ----
    gen = torch.Generator(device=device).manual_seed(a.seed)
    cgen = torch.Generator().manual_seed(a.seed)     # crop offsets
    perm, ptr = torch.randperm(n_seq, device=device, generator=gen), 0

    budget = a.minutes * 60.0
    train_time, step, seen = 0.0, 0, 0
    loss_acc, loss_n, last_log = torch.zeros((), device=device), 0, 0.0
    next_eval = a.eval_every * 60.0 if (a.eval_every and a.holdout) else None
    if not a.bench:
        show(f"{'min':>5s} {'passes':>6s} {'len':>5s} {'loss':>7s}" + (f" {'val acc':>8s}" if next_eval else ""))
    bench_marks = {}
    seg_start = time.time()
    while True:
        elapsed = train_time + (time.time() - seg_start)
        if (a.bench and step >= a.bench) or (not a.bench and elapsed >= budget):
            break
        if ptr + a.bs > n_seq:
            perm, ptr = torch.randperm(n_seq, device=device, generator=gen), 0
        xb = xtr[perm[ptr:ptr + a.bs]]
        ptr += a.bs
        cur_len = seq_len
        if crop_on and not a.bench and elapsed < a.crop_frac * budget:
            cur_len = a.crop_len
            o = int(torch.randint(0, seq_len - cur_len + 1, (1,), generator=cgen))
            xb = xb[:, o:o + cur_len]
        lr = lr_at(elapsed / budget, step)
        dcy = min(a.ema, (1 + step) / (10 + step))
        loss = train_step(xb, lr, dcy)
        step += 1
        seen += a.bs
        loss_acc += loss
        loss_n += 1

        if a.bench:
            if step in (10, a.bench):
                if cuda:
                    torch.cuda.synchronize()
                bench_marks[step] = time.time() - seg_start
            continue
        now = train_time + (time.time() - seg_start)
        if now - last_log >= a.log_every:
            row = f"{now / 60:5.1f} {seen / N_FINAL:6.1f} {cur_len:5d} {(loss_acc / loss_n).item():7.3f}"
            if next_eval is not None and now >= next_eval:
                if cuda:
                    torch.cuda.synchronize()
                train_time += time.time() - seg_start          # pause the training clock during the check
                acc, _ = evaluate(ema, xva[:300], device)
                row += f" {acc * 100:7.1f}%"
                while next_eval <= now:
                    next_eval += a.eval_every * 60.0
                seg_start = time.time()
            show(row)
            last_log, loss_acc, loss_n = now, torch.zeros((), device=device), 0

    if a.bench:
        first10 = bench_marks.get(10, 0.0)
        ms = (bench_marks[a.bench] - first10) / max(1, a.bench - 10) * 1000
        print(f"BENCH compile={a.compile} first10_s={first10:.2f} ms_per_step={ms:.2f} "
              f"eager_fallback={int(train_model is model and a.compile != 'none')}", flush=True)
        return

    if cuda:
        torch.cuda.synchronize()
    train_time += time.time() - seg_start
    torch.save({k: v.detach().cpu().float() for k, v in ema.state_dict().items()}, a.out)
    print(f"DONE minutes={train_time / 60:.2f} passes={seen / N_FINAL:.2f} steps={step} "
          f"compiled={int(train_model is not model)} out={a.out}", flush=True)
    if a.holdout:
        acc, ce = evaluate(ema, xva, device)
        print(f"RESULT acc={acc:.6f} ce={ce:.6f} steps={step} passes={seen / N_FINAL:.3f} "
              f"minutes={train_time / 60:.3f}", flush=True)


if __name__ == "__main__":
    main()
