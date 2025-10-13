#!/usr/bin/env python3
import argparse
from pathlib import Path
from textwrap import indent

import torch


def summarize_tensor(t):
    if isinstance(t, torch.Tensor):
        return f"Tensor(dtype={t.dtype}, shape={tuple(t.shape)})"
    return type(t).__name__


def summarize_state_dict(sd, max_items=10):
    lines = []
    for idx, (k, v) in enumerate(sd.items()):
        prefix = f"{idx:03d}: {k}"
        suffix = ""
        if isinstance(v, torch.Tensor):
            suffix = f" -> {v.dtype} {tuple(v.shape)}"
        elif isinstance(v, dict):
            suffix = f" -> dict({len(v)})"
        elif isinstance(v, list):
            suffix = f" -> list({len(v)})"
        else:
            suffix = f" -> {type(v).__name__}"
        lines.append(prefix + suffix)
        if idx + 1 == max_items:
            lines.append(f"... ({len(sd) - max_items} more entries)")
            break
    return "\n".join(lines)


def main(path: Path, max_print: int):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: {path}")
    print(f"Keys: {list(ckpt.keys())}\n")

    if "model" in ckpt:
        model_sd = ckpt["model"]
        print(f"[model] ({len(model_sd)} parameters)")
        print(indent(summarize_state_dict(model_sd, max_print), "  "))
        print()

    if "optimizers" in ckpt:
        optim_sd = ckpt["optimizers"]
        print(f"[optimizers] ({len(optim_sd)} entries)")
        for name, state in optim_sd.items():
            print(f"  {name}: state_dict keys {list(state.keys())}")
        print()

    if "schedulers" in ckpt:
        sched_sd = ckpt["schedulers"]
        print(f"[schedulers] ({len(sched_sd)} entries)")
        for name, state in sched_sd.items():
            print(f"  {name}: state_dict keys {list(state.keys())}")
        print()

    for meta_key in ("global_step", "epoch"):
        if meta_key in ckpt:
            print(f"{meta_key}: {ckpt[meta_key]}")

    if "model" in ckpt:
        total_params = sum(p.numel() for p in ckpt["model"].values() if isinstance(p, torch.Tensor))
        print(f"Total parameter count: {total_params:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a Flow3D checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to *.ckpt file")
    parser.add_argument("--max-items", type=int, default=20, help="Max model entries to print")
    args = parser.parse_args()
    main(args.checkpoint, args.max_items)


'''
python check_out_last_ckpt.py /home/ubuntu/deform-colon/shape-of-motion/OUTPUT_bootstapir/v4/checkpoints/last.ckpt --max-items 10

'''
