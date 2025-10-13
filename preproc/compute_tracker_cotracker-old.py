#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate CoTracker tracks in Shape-of-Motion file layout.

For a sequence of T frames:
- Run CoTracker once to get pred_tracks [1,T,N,2] and pred_visibility [1,T,N,1].
- For each query frame q, select the subset of points visible at q (vis>thresh).
- For every target frame t, write:  <out_dir>/<q_name>_<t_name>.npy  with shape (P,4):
    [x_px, y_px, occlusion_logit, expected_dist_logit]
  matching the TAPIR-style convention consumed by the dataset loader.
This matches the loader that stacks along the time axis to form (P, T, 4).

Typical use for C3VD:
    python make_cotracker_tracks_for_som.py \
        --frames_glob "/path/to/seq/rgb/*.png" \
        --out_dir "/path/to/seq/tracks/cotracker" \
        --grid_size 24 \
        --vis_thresh 0.5 \
        --tau_visible 1.5 --M_occluded 1000.0
"""

from __future__ import annotations
import os, re, glob, math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import imageio.v2 as imageio
from tqdm import tqdm

# CoTracker API (see README/demo)
# pred_tracks [B,T,N,2], pred_visibility [B,T,N,1]
# https://github.com/facebookresearch/co-tracker
from cotracker.predictor import CoTrackerPredictor  

import tyro


def natural_key(p: str) -> Tuple:
    """Sort filenames in human/numeric order."""
    s = str(Path(p).stem)
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s))


def load_frames(frames_glob: str, max_frames: int | None = None) -> Tuple[np.ndarray, List[str]]:
    paths = sorted(glob.glob(frames_glob), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No frames matched: {frames_glob}")
    if max_frames is not None:
        paths = paths[:max_frames]

    imgs = []
    for p in paths:
        im = imageio.imread(p)
        if im.ndim == 2:
            im = np.stack([im]*3, axis=-1)
        if im.shape[-1] == 4:
            im = im[..., :3]
        imgs.append(im)
    arr = np.stack(imgs, axis=0)  # [T,H,W,3], uint8
    return arr, [Path(p).stem for p in paths]


def to_video_tensor(frames_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """[T,H,W,3] uint8 -> [1,T,3,H,W] float32 in [0,1]."""
    T, H, W, _ = frames_uint8.shape
    vid = torch.from_numpy(frames_uint8).to(device=device, dtype=torch.float32) / 255.0
    vid = vid.permute(0, 3, 1, 2).unsqueeze(0)  # [1,T,3,H,W]
    return vid


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def select_points_for_query(pred_visibility: torch.Tensor,
                            q: int,
                            vis_thresh: float,
                            min_points: int,
                            topk_window: int) -> torch.Tensor:
    """
    Choose indices of points to anchor at query q.
    pred_visibility: [T,N] (after squeeze); returns indices [P] for which:
      - primarily vis[q,n] > vis_thresh; if none, fallback to top-K by mean vis in a window.
    """
    T, N = pred_visibility.shape
    vis_q = pred_visibility[q]  # [N]
    good = (vis_q > vis_thresh).nonzero(as_tuple=False).flatten()
    if good.numel() >= min_points:
        return good

    # Fallback: rank by mean visibility in a temporal window around q
    half = max(0, topk_window // 2)
    left = max(0, q - half)
    right = min(T, q + half + 1)
    mean_vis = pred_visibility[left:right].mean(0)  # [N]
    k = min(max(min_points, good.numel()), N)
    _, idx = torch.topk(mean_vis, k, largest=True)
    return idx


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)

def save_pair_array(out_path: Path,
                    xy_t: np.ndarray,   # [P,2] float or convertible
                    vis_t: np.ndarray,  # [P]   visibility prob in [0,1]
                    vis_thresh: float,  # below this ⇒ hard-occluded
                    tau_visible: float, # temperature (<1 sharper, >1 softer)
                    M_occluded: float) -> None:
    """
    Save TAPIR-style (P,4) array with columns:
        [x_px, y_px, occlusion_logit, expected_dist_logit]

    Conventions:
      - occlusion probability p_occ = 1 - visibility
      - logits are temperature-scaled by tau_visible (divide the logit)
      - if visibility < vis_thresh, both logits are set to +M_occluded
      - expected_dist_logit duplicates occlusion_logit (SOM/TAPIR-compatible)
    """
    eps = 1e-6
    xy = np.asarray(xy_t, dtype=np.float32)
    vis = np.asarray(vis_t, dtype=np.float32)

    # Basic shape checks (lightweight; remove if you prefer)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy_t must be (P,2), got {xy.shape}")
    if vis.ndim != 1 or vis.shape[0] != xy.shape[0]:
        raise ValueError(f"vis_t must be (P,), got {vis.shape}; P must match xy_t")

    # Sanitize visibility (replace NaNs with 0 → treat as occluded)
    vis = np.where(np.isfinite(vis), vis, 0.0)
    vis = np.clip(vis, 0.0, 1.0)

    # Base occlusion probability and logit
    p_occ = np.clip(1.0 - vis, eps, 1.0 - eps)
    occ_logit = _logit(p_occ)

    # Temperature scaling: smaller tau ⇒ sharper (bigger magnitude) logits
    tau = max(float(tau_visible), eps)
    occ_logit = occ_logit / tau

    # Expected-distance channel (kept identical for TAPIR/SOM compatibility)
    ed_logit = occ_logit.copy()

    # Hard occlusion override for low-visibility points
    hard_mask = vis < float(vis_thresh)
    if np.any(hard_mask):
        occ_logit[hard_mask] = float(M_occluded)
        ed_logit[hard_mask]  = float(M_occluded)

    packed = np.concatenate(
        [xy, occ_logit[:, None].astype(np.float32), ed_logit[:, None].astype(np.float32)],
        axis=1
    ).astype(np.float32)

    np.save(out_path, packed)

def main(
    frames_glob: str,
    out_dir: str,
    grid_size: int = 32,            # CoTracker quasi-dense grid size (≈ grid_size x grid_size points)
    checkpoint: str | None = None,  # if None, CoTrackerPredictor loads default
    device_str: str = "cuda",
    vis_thresh: float = 0.5,
    tau_visible: float = 1.5,
    M_occluded: float = 1000.0,
    min_points: int = 64,
    topk_window: int = 11,
    max_frames: int | None = None,
) -> None:

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    # 1) Load frames and names
    frames_u8, frame_names = load_frames(frames_glob, max_frames=max_frames)  # [T,H,W,3]
    T, H, W, _ = frames_u8.shape
    print(f"[info] loaded {T} frames @ {W}x{H}")

    # 2) Build video tensor
    video = to_video_tensor(frames_u8, device)  # [1,T,3,H,W]

    # 3) Run CoTracker in grid mode
    #    Returns pred_tracks [1,T,N,2] (pixels), pred_visibility [1,T,N,1] (0..1).
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, grid_size=grid_size)  # pixels, prob
    pred_tracks = pred_tracks[0].detach().cpu()  # [T,N,2]

    pred_visibility = pred_visibility[0].detach().cpu().to(torch.float32)  # [T,N] or [T,N,1]
    if pred_visibility.ndim == 3:
        if pred_visibility.shape[-1] != 1:
            raise ValueError(
                "Unexpected visibility tensor shape "
                f"{tuple(pred_visibility.shape)}; expected trailing size 1."
            )
        pred_visibility = pred_visibility[..., 0]

    # 4) Prepare output directory
    out_root = Path(out_dir)
    ensure_dir(out_root)

    q = 0
    idx_q = select_points_for_query(pred_visibility, q, vis_thresh, min_points, topk_window)  # [P]
    P = int(idx_q.numel())
    if P == 0:
        print(f"[warn] q={q} has zero selected points; skipping.")
        continue

    # Extract per-t arrays once (avoid slicing twice)
    # xy_all_t: list of (P,2); vis_all_t: list of (P,)
    xy_all_t = pred_tracks[:, idx_q, :].numpy()     # [T,P,2]
    vis_all_t = pred_visibility[:, idx_q].numpy()   # [T,P]

    q_name = frame_names[q]
    for t in range(T):
        t_name = frame_names[t]
        out_path = out_root / f"{q_name}_{t_name}.npy"
        xy_t = xy_all_t[t]
        if t == q:
            xy_t = np.rint(xy_t)
            xy_t[..., 0] = np.clip(xy_t[..., 0], 0, W - 1)
            xy_t[..., 1] = np.clip(xy_t[..., 1], 0, H - 1)
        save_pair_array(out_path, xy_t, vis_all_t[t], vis_thresh, tau_visible, M_occluded)

    print(f"[done] Wrote pairwise track files to: {out_root}")


if __name__ == "__main__":
    if tyro is None:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--frames_glob", required=True, help="e.g., '/path/to/seq/rgb/*.png'")
        ap.add_argument("--out_dir", required=True, help="Output directory for <q>_<t>.npy files")
        ap.add_argument("--grid_size", type=int, default=24)
        ap.add_argument("--checkpoint", type=str, default=None)
        ap.add_argument("--device_str", type=str, default="cuda")
        ap.add_argument("--vis_thresh", type=float, default=0.5)
        ap.add_argument("--tau_visible", type=float, default=1.5)
        ap.add_argument("--M_occluded", type=float, default=1000.0)
        ap.add_argument("--min_points", type=int, default=64)
        ap.add_argument("--topk_window", type=int, default=11)
        ap.add_argument("--max_frames", type=int, default=None)
        args = ap.parse_args()
        main(**vars(args))
    else:
        tyro.cli(main)
