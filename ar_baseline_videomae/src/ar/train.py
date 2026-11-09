"""EPIC-KITCHENS-100 action recognition baseline (VideoMAE + verb/noun heads).

Run from the repo root:
    python -m src.ar.train --train_csv ... --val_csv ... --video_root ... --out_dir ...

Re-running the same command resumes from <out_dir>/last.pt automatically.
Outputs in out_dir: config.json, metrics.jsonl, last.pt, best.pt, best_metrics.json,
(and preds_val.csv / preds_val.json with --export / --export_json).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import ClipDataset
from .metrics import Meter
from .model import DEFAULT_MODEL, VideoMAEVerbNoun
from .utils import load_annotations


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--train_csv")
    p.add_argument("--val_csv", required=True)
    p.add_argument("--video_root", required=True, help="folder with P01/P01_01.MP4 ...")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--skip_missing", action="store_true", help="drop clips whose video file is missing")
    p.add_argument("--n_verb", type=int, default=97)
    p.add_argument("--n_noun", type=int, default=300)
    # model
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF name, or 'tiny-random' for a smoke test")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--crop", type=int, default=224)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (saves memory)")
    p.add_argument("--dropout", type=float, default=0.1)
    # optimisation
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=5e-5, help="backbone lr")
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--aug", choices=["basic", "none"], default="basic")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=20)
    # debugging / modes
    p.add_argument("--overfit", type=int, default=0, help="train AND validate on the first N train clips")
    p.add_argument("--max_val", type=int, default=0, help="only use the first N validation clips")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--ckpt", help="checkpoint for --eval_only (default: <out_dir>/best.pt)")
    p.add_argument("--export", action="store_true", help="write preds_val.csv after (final) evaluation")
    p.add_argument("--export_json", action="store_true", help="also write EPIC-style preds_val.json")
    return p.parse_args()


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def lr_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))


def make_groups(model, lr, head_lr, wd):
    groups = []
    for params, base in ((model.backbone_params(), lr), (model.head_params(), head_lr)):
        decay = [p for p in params if p.requires_grad and p.ndim > 1]
        no_decay = [p for p in params if p.requires_grad and p.ndim <= 1]
        if decay:
            groups.append({"params": decay, "lr": base, "base_lr": base, "weight_decay": wd})
        if no_decay:
            groups.append({"params": no_decay, "lr": base, "base_lr": base, "weight_decay": 0.0})
    return groups


def make_loader(ds, args, dev, shuffle, gen=None):
    bs = args.batch_size if shuffle else args.batch_size * 2
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=dev.type == "cuda", drop_last=shuffle and len(ds) >= bs,
                      generator=gen)


@torch.no_grad()
def evaluate(model, loader, dev, amp_dtype, collect=False):
    model.eval()
    meter, store = Meter(), []
    for x, v, n, idx in loader:
        x = x.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            vl, nl = model(x)
        vl, nl = vl.float().cpu(), nl.float().cpu()
        meter.update(vl, nl, v, n)
        if collect:
            store.append((idx, vl, nl))
    return meter.compute(), store


def export_predictions(df, store, out_dir, export_json):
    idx = torch.cat([s[0] for s in store])
    order = idx.argsort()
    idx = idx[order]
    vl = torch.cat([s[1] for s in store])[order]
    nl = torch.cat([s[2] for s in store])[order]
    vp, npr = vl.softmax(1), nl.softmax(1)
    joint = (vp[:, :, None] * npr[:, None, :]).flatten(1)
    top_s, top_i = joint.topk(100, dim=1)
    n_noun = nl.shape[1]
    sub = df.iloc[idx.numpy()].reset_index(drop=True)
    rows = pd_rows(sub, vl, nl, top_i, n_noun)
    pd.DataFrame(rows).to_csv(out_dir / "preds_val.csv", index=False)
    if export_json:
        results = {}
        for r, nid in enumerate(sub["narration_id"]):
            results[nid] = {
                "verb": {str(k): float(vp[r, k]) for k in range(vp.shape[1])},
                "noun": {str(k): float(npr[r, k]) for k in range(npr.shape[1])},
                "action": {f"{int(i) // n_noun},{int(i) % n_noun}": float(s)
                           for i, s in zip(top_i[r], top_s[r])},
            }
        blob = {"version": "0.2", "challenge": "action_recognition",
                "sls_pt": 2, "sls_tl": 3, "sls_td": 3, "results": results}
        with open(out_dir / "preds_val.json", "w") as f:
            json.dump(blob, f)
    print(f"exported predictions for {len(sub)} clips to {out_dir}", flush=True)


def pd_rows(sub, vl, nl, top_i, n_noun):
    v5 = vl.topk(5, 1).indices
    n5 = nl.topk(5, 1).indices
    rows = []
    for r in range(len(sub)):
        a5 = [f"{int(i) // n_noun},{int(i) % n_noun}" for i in top_i[r, :5]]
        row = {"narration_id": sub["narration_id"].iloc[r],
               "verb_pred": int(v5[r, 0]), "noun_pred": int(n5[r, 0]), "action_pred": a5[0],
               "verb_top5": " ".join(str(int(k)) for k in v5[r]),
               "noun_top5": " ".join(str(int(k)) for k in n5[r]),
               "action_top5": " ".join(a5)}
        if int(sub["verb_class"].iloc[r]) >= 0:
            row["verb_true"] = int(sub["verb_class"].iloc[r])
            row["noun_true"] = int(sub["noun_class"].iloc[r])
        if "narration" in sub.columns:
            row["narration"] = sub["narration"].iloc[r]
        rows.append(row)
    return rows


def save_ckpt(path, model, opt, scaler, epoch, gstep, best, args):
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "global_step": gstep, "best": best, "args": vars(args)}, tmp)
    os.replace(tmp, path)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if dev.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"device={dev} amp={amp_dtype}", flush=True)

    ck = None
    if args.eval_only:  # rebuild exactly the architecture that was trained
        ck_path = args.ckpt or str(out / "best.pt")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        for k in ("model", "frames", "crop", "n_verb", "n_noun"):
            setattr(args, k, ck["args"][k])
        print(f"eval_only: checkpoint {ck_path} (epoch {ck['epoch']})", flush=True)
    set_seed(args.seed)

    va_df = load_annotations(args.val_csv)
    tr_df = None
    if not args.eval_only:
        if not args.train_csv:
            raise SystemExit("--train_csv is required for training")
        tr_df = load_annotations(args.train_csv)
        if args.overfit:
            tr_df = tr_df.head(args.overfit).reset_index(drop=True)
            va_df = tr_df.copy()
            args.aug = "none"
    if args.max_val:
        va_df = va_df.head(args.max_val).reset_index(drop=True)

    for name, df in (("train", tr_df), ("val", va_df)):
        if df is not None and (df["verb_class"].max() >= args.n_verb or df["noun_class"].max() >= args.n_noun):
            raise SystemExit(f"{name}: class id out of range (check --n_verb/--n_noun)")

    va_ds = ClipDataset(va_df, args.video_root, args.frames, False, args.crop, "none", args.seed, args.skip_missing)
    va_loader = make_loader(va_ds, args, dev, shuffle=False)

    model = VideoMAEVerbNoun(args.model, args.frames, args.n_verb, args.n_noun, args.dropout,
                             args.freeze_backbone, args.grad_ckpt).to(dev)

    if args.eval_only:
        model.load_state_dict(ck["model"])
        metrics, store = evaluate(model, va_loader, dev, amp_dtype, collect=args.export or args.export_json)
        print(json.dumps(metrics, indent=1), flush=True)
        (out / "metrics_eval.json").write_text(json.dumps(metrics, indent=1))
        if store:
            export_predictions(va_ds.df, store, out, args.export_json)
        return

    tr_ds = ClipDataset(tr_df, args.video_root, args.frames, True, args.crop, args.aug, args.seed, args.skip_missing)
    gen = torch.Generator()
    tr_loader = make_loader(tr_ds, args, dev, shuffle=True, gen=gen)
    steps_per_epoch = max(1, len(tr_loader) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    print(f"train clips={len(tr_ds)} val clips={len(va_ds)} steps/epoch={steps_per_epoch} "
          f"effective batch={args.batch_size * args.accum}", flush=True)

    opt = torch.optim.AdamW(make_groups(model, args.lr, args.head_lr, args.wd), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    start_epoch, gstep, best = 0, 0, -1.0
    last = out / "last.pt"
    if last.exists() and not args.no_resume:
        c = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        scaler.load_state_dict(c["scaler"])
        start_epoch, gstep, best = c["epoch"], c["global_step"], c["best"]
        print(f"RESUMED from {last}: {start_epoch} epochs done, best action@1={best:.2f}", flush=True)
    else:
        (out / "config.json").write_text(json.dumps({**vars(args), "git_commit": git_commit()}, indent=1))

    for epoch in range(start_epoch, args.epochs):
        tr_ds.set_epoch(epoch)
        gen.manual_seed(args.seed * 1000 + epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        t0, loss_sum, n_it = time.time(), 0.0, 0
        for it, (x, v, n, _) in enumerate(tr_loader):
            x, v, n = x.to(dev, non_blocking=True), v.to(dev), n.to(dev)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                vl, nl = model(x)
            loss = (F.cross_entropy(vl.float(), v, label_smoothing=args.label_smoothing)
                    + F.cross_entropy(nl.float(), n, label_smoothing=args.label_smoothing))
            scaler.scale(loss / args.accum).backward()
            loss_sum += float(loss)
            n_it += 1
            if (it + 1) % args.accum == 0:
                f = lr_factor(gstep, warmup_steps, total_steps)
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * f
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                gstep += 1
            if (it + 1) % args.log_every == 0:
                print(f"ep {epoch + 1}/{args.epochs} it {it + 1}/{len(tr_loader)} "
                      f"loss {loss_sum / n_it:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{(time.time() - t0) / n_it:.2f}s/it", flush=True)

        metrics, _ = evaluate(model, va_loader, dev, amp_dtype)
        rec = {"epoch": epoch + 1, "train_loss": loss_sum / max(1, n_it), "val": metrics,
               "minutes": (time.time() - t0) / 60}
        print(json.dumps(rec), flush=True)
        with open(out / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if metrics.get("action_top1", -1) > best:
            best = metrics["action_top1"]
            save_ckpt(out / "best.pt", model, opt, scaler, epoch + 1, gstep, best, args)
            (out / "best_metrics.json").write_text(json.dumps({"epoch": epoch + 1, **metrics}, indent=1))
        save_ckpt(last, model, opt, scaler, epoch + 1, gstep, best, args)

    if args.export or args.export_json:
        model.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=False)["model"])
        _, store = evaluate(model, va_loader, dev, amp_dtype, collect=True)
        export_predictions(va_ds.df, store, out, args.export_json)
    print("done", flush=True)


if __name__ == "__main__":
    main()
