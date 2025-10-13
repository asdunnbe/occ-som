import argparse
import glob
import os
from pathlib import Path

import imageio.v2 as imageio
import mediapy as media
import numpy as np
import torch
from cotracker.predictor import CoTrackerPredictor
from tqdm import tqdm


def read_video(folder_path: str, force_rgb: bool = False) -> np.ndarray:
    """Load an image sequence (sorted lexicographically) into a numpy array."""
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in directory: {folder_path}")

    frames = []
    for frame_path in frame_paths:
        frame = imageio.imread(frame_path)
        if force_rgb:
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[..., :3]
            elif frame.shape[-1] != 3:
                raise ValueError(f"Unsupported number of channels in {frame_path}: {frame.shape}")
        frames.append(frame)
    return np.stack(frames)


def ensure_mask_files(mask_dir: str, frame_names: list[str], height: int, width: int) -> None:
    """Ensure every frame has a corresponding binary mask file; fill gaps with all-ones masks."""
    os.makedirs(mask_dir, exist_ok=True)
    existing = {os.path.basename(path): path for path in glob.glob(os.path.join(mask_dir, "*"))}
    if len(existing) == len(frame_names):
        return

    full_mask = np.ones((height, width), dtype=np.uint8)
    for name in frame_names:
        if name not in existing:
            imageio.imwrite(os.path.join(mask_dir, name), full_mask)


def video_to_tensor(frames_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert uint8 video [T, H, W, 3] to float tensor [1, T, 3, H, W] in [0, 1]."""
    tensor = torch.from_numpy(frames_uint8).to(device=device, dtype=torch.float32) / 255.0
    tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    return tensor


def compute_occ_logits(visibility: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Convert visibility probabilities into TAPIR-style occlusion / expected-dist logits."""
    vis = np.clip(visibility.astype(np.float32), eps, 1.0 - eps)
    occ_prob = np.clip(1.0 - vis, eps, 1.0 - eps)
    logits = np.log(occ_prob) - np.log(1.0 - occ_prob)
    return logits.astype(np.float32)


def maybe_resize_video(video: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resize video if target dimensions are provided."""
    if target_height <= 0 or target_width <= 0:
        return video
    if video.shape[1] == target_height and video.shape[2] == target_width:
        return video
    resized = media.resize_video(video, (target_height, target_width))
    resized = np.clip(np.rint(resized), 0, 255).astype(np.uint8)
    return resized


def rescale_tracks(
    tracks: np.ndarray,
    orig_width: int,
    orig_height: int,
    resized_width: int,
    resized_height: int,
) -> np.ndarray:
    """Map tracks predicted on resized frames back to the original resolution."""
    scaled = tracks.astype(np.float32, copy=True)
    if resized_width > 1 and orig_width > 1 and resized_width != orig_width:
        scaled[..., 0] *= (orig_width - 1) / (resized_width - 1)
    if resized_height > 1 and orig_height > 1 and resized_height != orig_height:
        scaled[..., 1] *= (orig_height - 1) / (resized_height - 1)
    return scaled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with RGB frames.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory with binary masks.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for <query>_<target>.npy files.")
    parser.add_argument(
        "--grid_size",
        type=int,
        default=4,
        help="Number of grid points per image axis for seeding CoTracker queries.",
    )
    parser.add_argument("--resize_height", type=int, default=256, help="Optional preprocessing resize height.")
    parser.add_argument("--resize_width", type=int, default=256, help="Optional preprocessing resize width.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (defaults to CUDA when available).",
    )
    parser.add_argument(
        "--points_per_chunk",
        type=int,
        default=128,
        help="Number of query points processed per CoTracker forward pass.",
    )
    args = parser.parse_args()

    image_dir = args.image_dir
    mask_dir = args.mask_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    if not frame_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    frame_names = [os.path.basename(path) for path in frame_paths]

    done = True
    for frame_name in frame_names:
        prefix = os.path.splitext(frame_name)[0]
        existing = glob.glob(os.path.join(out_dir, f"{prefix}_*.npy"))
        if len(existing) != len(frame_names):
            done = False
            break
    print(f"{done=}")
    if done:
        print("All CoTracker pair files already present. Skipping.")
        return

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    model.eval()

    video = read_video(image_dir, force_rgb=True)  # [T, H, W, 3], uint8
    num_frames, height, width = video.shape[0:3]
    print(f"Loaded video: {video.shape}, dtype={video.dtype}")

    ensure_mask_files(mask_dir, frame_names, height, width)

    mask_video = read_video(mask_dir)  # allow grayscale or rgb masks
    masks = (mask_video.reshape(num_frames, height, width, -1) > 0).any(axis=-1)
    print(f"Masks: {masks.shape}, valid pixels={masks.sum()}")

    resize_height = args.resize_height if args.resize_height > 0 else height
    resize_width = args.resize_width if args.resize_width > 0 else width
    resized_video = maybe_resize_video(video, resize_height, resize_width)
    print(f"Resized video: {resized_video.shape}")

    video_tensor = video_to_tensor(resized_video, device)

    grid_points = max(1, int(args.grid_size))
    y_lin = np.linspace(0, height - 1, num=grid_points, dtype=np.float32)
    x_lin = np.linspace(0, width - 1, num=grid_points, dtype=np.float32)
    y_grid_float, x_grid_float = np.meshgrid(y_lin, x_lin, indexing="ij")

    y_grid_int = np.clip(np.rint(y_grid_float).astype(np.int32), 0, height - 1)
    x_grid_int = np.clip(np.rint(x_grid_float).astype(np.int32), 0, width - 1)

    if width > 1 and resize_width > 1:
        scale_x = (resize_width - 1) / (width - 1)
        x_grid_resized = x_grid_float * scale_x
    else:
        x_grid_resized = np.zeros_like(x_grid_float, dtype=np.float32)
    if height > 1 and resize_height > 1:
        scale_y = (resize_height - 1) / (height - 1)
        y_grid_resized = y_grid_float * scale_y
    else:
        y_grid_resized = np.zeros_like(y_grid_float, dtype=np.float32)

    for t in tqdm(range(num_frames), desc="query frames"):
        query_name = os.path.splitext(frame_names[t])[0]
        existing_pairs = glob.glob(os.path.join(out_dir, f"{query_name}_*.npy"))
        if len(existing_pairs) == num_frames:
            print(f"Skipping query frame {t} ({query_name}); already processed.")
            continue

        mask_t = masks[t]
        in_mask = mask_t[y_grid_int, x_grid_int]

        query_xy_orig = np.stack(
            [x_grid_int[in_mask].astype(np.float32), y_grid_int[in_mask].astype(np.float32)],
            axis=-1,
        )
        query_xy_resized = np.stack(
            [x_grid_resized[in_mask].astype(np.float32), y_grid_resized[in_mask].astype(np.float32)],
            axis=-1,
        )

        num_points = query_xy_orig.shape[0]
        print(f"query {t}: total grid={grid_points * grid_points}, in-mask={num_points}")

        if num_points == 0:
            outputs = np.zeros((0, num_frames, 4), dtype=np.float32)
        else:
            query_triplets = np.concatenate(
                [
                    np.full((num_points, 1), t, dtype=np.float32),
                    query_xy_resized[:, :1],  # x
                    query_xy_resized[:, 1:],  # y
                ],
                axis=1,
            )

            chunk_size = max(1, int(args.points_per_chunk))
            chunk_outputs = []
            for start_idx in tqdm(range(0, num_points, chunk_size), desc="points", leave=False):
                end_idx = min(start_idx + chunk_size, num_points)
                pts_chunk = query_triplets[start_idx:end_idx]
                pts_tensor = torch.from_numpy(pts_chunk).unsqueeze(0).to(device)

                with torch.inference_mode():
                    pred_tracks, pred_visibility = model(video_tensor, queries=pts_tensor)

                tracks_chunk = pred_tracks[0].detach().cpu().numpy()  # [T, chunk, 2]
                visibility_chunk = pred_visibility[0].detach().cpu().float().numpy()  # [T, chunk]

                tracks_chunk = np.transpose(tracks_chunk, (1, 0, 2))  # [chunk, T, 2]
                visibility_chunk = np.transpose(visibility_chunk, (1, 0))  # [chunk, T]

                tracks_chunk = rescale_tracks(
                    tracks_chunk, width, height, resize_width, resize_height
                )

                occ_logits = compute_occ_logits(visibility_chunk)  # [chunk, T]
                chunk_output = np.concatenate(
                    [
                        tracks_chunk,
                        occ_logits[..., None],
                        occ_logits[..., None],
                    ],
                    axis=-1,
                ).astype(np.float32)
                chunk_outputs.append(chunk_output)

            outputs = np.concatenate(chunk_outputs, axis=0) if chunk_outputs else np.zeros(
                (0, num_frames, 4), dtype=np.float32
            )

        if outputs.shape[0] > 0:
            outputs[:, t, :2] = query_xy_orig

        for j in range(num_frames):
            target_name = os.path.splitext(frame_names[j])[0]
            out_path = Path(out_dir) / f"{query_name}_{target_name}.npy"
            np.save(out_path, outputs[:, j])


if __name__ == "__main__":
    main()
