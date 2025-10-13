"""Utility to reshape raw C3VD exports into the Flow3D data layout.

The script copies RGB frames, converts depth maps into ``.npy`` blobs, fabricates
foreground masks filled with ones (no background present), carries over camera
metadata, and can optionally launch TAPIR/Bootstrapped tracks.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


import numpy as np
import tyro

try:
    import imageio.v2 as imageio
except Exception: 
    imageio = None

try:
    from PIL import Image
except Exception: 
    Image = None


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_DEPTH_EXTS = {".png", ".npy", ".npz", ".exr", ".tif", ".tiff"}
_DEPTH_PRIORITY = {".npy": 0, ".npz": 1, ".exr": 2, ".tif": 3, ".tiff": 4, ".png": 5}


def _sanitize_numeric_stem(stem: str) -> str:
    digits_only = "".join(ch for ch in stem if ch.isdigit())
    return digits_only if digits_only else stem


def _iter_files(root: Path, exts: set[str]) -> Iterable[Path]:
    for entry in sorted(root.iterdir()):
        if entry.is_file() and entry.suffix.lower() in exts:
            yield entry


def _dir_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _read_image(path: Path) -> np.ndarray:
    if imageio is not None:
        return imageio.imread(path)
    if Image is not None:
        with Image.open(path) as img:
            return np.array(img)
    raise RuntimeError("Install imageio or pillow to read images.")


def _write_png(path: Path, array: np.ndarray) -> None:
    if imageio is not None:
        imageio.imwrite(path, array)
        return
    if Image is not None:
        Image.fromarray(array).save(path)
        return
    raise RuntimeError("Install imageio or pillow to write images.")


def _write_mask(path: Path, shape: tuple[int, int], value: int = 1) -> None:
    mask = np.full(shape, value, dtype=np.uint8)
    _write_png(path, mask)


def _save_depth_npy(src: Path, dst: Path) -> tuple[int, int]:
    arr: np.ndarray
    if src.suffix.lower() == ".npy":
        arr = np.load(src)
    elif src.suffix.lower() == ".npz":
        with np.load(src) as data:
            if len(data.files) != 1:
                raise ValueError(f"Expected single array in {src}, found {data.files}.")
            arr = data[data.files[0]]
    else:
        arr = _read_image(src)
    arr = np.asarray(arr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, arr.astype(np.float32))
    return arr.shape[:2]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_split_json(path: Path, stems: list[str], indices: list[int]) -> None:
    payload = {
        "camera_ids": [0 for _ in indices],
        "frame_names": [stems[i] for i in indices],
        "time_ids": indices,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=4)


def _generate_splits(image_files: list[Path], output_dir: Path, stride: int = 8) -> None:
    if not image_files or stride <= 0:
        return
    stems = [p.stem for p in image_files]
    total = len(stems)
    stride = max(stride, 1)
    val_indices = list(range(0, total, stride))
    val_set = set(val_indices)
    train_indices = [idx for idx in range(total) if idx not in val_set]
    splits_dir = output_dir / "splits"
    _write_split_json(splits_dir / "train.json", stems, train_indices)
    _write_split_json(splits_dir / "val.json", stems, val_indices)


def _infer_frames_glob(image_dir: Path, image_files: list[Path]) -> str:
    """Return a glob string that matches the copied frames inside image_dir."""
    suffixes = {path.suffix.lower() for path in image_files if path.suffix}
    suffix = suffixes.pop() if len(suffixes) == 1 else None
    pattern = f"*{suffix}" if suffix else "*"
    return str((image_dir / pattern).resolve())


def _run_tracks(
    image_dir: Path,
    mask_dir: Path,
    track_dir: Path,
    track_model: str,
    tapir_torch: bool,
    gpu: int | None,
    grid_size: int,
    image_files: list[Path],
) -> None:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    script_dir = Path(__file__).resolve().parent
    track_model_lower = track_model.lower()

    if track_model_lower == "cotracker":
        cmd = [
            "python",
            str(script_dir / "compute_tracks_cotracker.py"),
            "--image_dir",
            str(image_dir),
            "--mask_dir",
            str(mask_dir),
            "--out_dir",
            str(track_dir),
            "--grid_size",
            str(grid_size),
        ]
    else:
        track_script = "compute_tracks_torch.py" if tapir_torch else "compute_tracks_jax.py"
        cmd = [
            "python",
            str(script_dir / track_script),
            "--image_dir",
            str(image_dir),
            "--mask_dir",
            str(mask_dir),
            "--out_dir",
            str(track_dir),
            "--model_type",
            track_model,
            "--grid_size",
            str(grid_size),
        ]

    cmd_str = " ".join(cmd)
    if gpu is not None:
        cmd_str = f"CUDA_VISIBLE_DEVICES={gpu} {cmd_str}"
    print("\n\n", cmd_str, "\n\n")
    subprocess.run(cmd, check=True, env=env)


def process_c3vd(
    input_dir: Path,
    output_dir: Path,
    image_subdir: str = "image/rgb",
    depth_subdir: str = "depth",
    image_out_name: str = "images",
    depth_out_name: str = "depths",
    mask_out_name: str = "masks",
    mask_value: int = 1,
    pose_name: str = "pose.txt",
    intrinsics_name: str = "intrinsics.json",
    track_out_name: str = "bootstapir",
    track_model: str = "bootstapir",
    tapir_torch: bool = True,
    gpu: int | None = None,
    run_tracks: bool = True,
    grid_size: int = 16,
    split_stride: int = 8,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_src = input_dir / image_subdir
    depth_src = input_dir / depth_subdir
    if not image_src.exists():
        raise FileNotFoundError(f"Missing image source directory: {image_src}")
    if not depth_src.exists():
        raise FileNotFoundError(f"Missing depth source directory: {depth_src}")

    image_dst = output_dir / image_out_name
    depth_dst = output_dir / depth_out_name
    mask_dst = output_dir / mask_out_name
    track_dst = output_dir / track_out_name

    image_dst.mkdir(parents=True, exist_ok=True)
    depth_dst.mkdir(parents=True, exist_ok=True)
    mask_dst.mkdir(parents=True, exist_ok=True)
    if run_tracks:
        track_dst.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(input_dir / pose_name, output_dir / pose_name)
    _copy_if_exists(input_dir / intrinsics_name, output_dir / intrinsics_name)

    image_files = list(_iter_files(image_src, _IMAGE_EXTS))
    if not image_files:
        raise ValueError(f"No image files found in {image_src}")

    depth_files_raw = list(_iter_files(depth_src, _DEPTH_EXTS))
    if not depth_files_raw:
        raise ValueError(f"No depth files found in {depth_src}")

    _generate_splits(image_files, output_dir, split_stride)

    if mask_value not in (0, 1):
        raise ValueError("mask_value must be 0 or 1 for binary masks")

    depth_files: dict[str, Path] = {}
    for depth_path in depth_files_raw:
        stem = depth_path.stem
        prev = depth_files.get(stem)
        if prev is None:
            depth_files[stem] = depth_path
            continue
        prev_score = _DEPTH_PRIORITY.get(prev.suffix.lower(), 100)
        new_score = _DEPTH_PRIORITY.get(depth_path.suffix.lower(), 100)
        if new_score < prev_score:
            depth_files[stem] = depth_path

    images_full = all((image_dst / img.name).exists() for img in image_files)
    if images_full:
        print(f"Image folder already populated, skipping copy: {image_dst}")
    else:
        for img_path in image_files:
            dst_path = image_dst / img_path.name
            if dst_path.exists():
                continue
            shutil.copy2(img_path, dst_path)

    depth_full = all(
        (depth_dst / f"{_sanitize_numeric_stem(stem)}.npy").exists()
        for stem in depth_files
    )
    if depth_full:
        print(f"Depth folder already populated, skipping conversion: {depth_dst}")
    else:
        for stem, depth_path in depth_files.items():
            sanitized_stem = _sanitize_numeric_stem(stem)
            dst_path = depth_dst / f"{sanitized_stem}.npy"
            if dst_path.exists():
                continue
            _save_depth_npy(depth_path, dst_path)

    mask_full = all((mask_dst / f"{img.stem}.png").exists() for img in image_files)
    if mask_full:
        print(f"Mask folder already populated, skipping generation: {mask_dst}")
    else:
        for img_path in image_files:
            mask_path = mask_dst / f"{img_path.stem}.png"
            if mask_path.exists():
                continue
            reference_path = image_dst / img_path.name
            if not reference_path.exists():
                reference_path = img_path
            shape = _read_image(reference_path).shape[:2]
            _write_mask(mask_path, shape, mask_value)

    if run_tracks:
        if _dir_has_files(track_dst):
            print(f"Track folder already populated, skipping tracking: {track_dst}")
        else:
            _run_tracks(
                image_dst,
                mask_dst,
                track_dst,
                track_model,
                tapir_torch,
                gpu,
                grid_size,
                image_files,
            )
        # _run_tracks(
        #     image_dst,
        #     mask_dst,
        #     track_dst,
        #     track_model,
        #     tapir_torch,
        #     gpu,
        #     grid_size,
        #     image_files,
        # )



def main(
    input_dir: str,
    output_dir: str,
    image_subdir: str = "rgb",
    depth_subdir: str = "depth",
    image_out_name: str = "images",
    depth_out_name: str = "depths",
    mask_out_name: str = "masks",
    mask_value: int = 1,
    pose_name: str = "pose.txt",
    intrinsics_name: str = "intrinsics.json",
    track_out_name: str = "bootstapir",
    track_model: str = "bootstapir",
    tapir_torch: bool = True,
    gpu: int | None = None,
    run_tracks: bool = True,
    grid_size: int = 16,
    split_stride: int = 8,
) -> None:
    process_c3vd(
        Path(input_dir).expanduser(),
        Path(output_dir).expanduser(),
        image_subdir=image_subdir,
        depth_subdir=depth_subdir,
        image_out_name=image_out_name,
        depth_out_name=depth_out_name,
        mask_out_name=mask_out_name,
        mask_value=mask_value,
        pose_name=pose_name,
        intrinsics_name=intrinsics_name,
        track_out_name=track_out_name,
        track_model=track_model,
        tapir_torch=tapir_torch,
        gpu=gpu,
        run_tracks=run_tracks,
        grid_size=grid_size,
        split_stride=split_stride,
    )


if __name__ == "__main__":
    tyro.cli(main)


'''
python process_c3vd.py --input_dir /home/ubuntu/deform-colon/data/v4 \
--output_dir /home/ubuntu/deform-colon/shape-of-motion/data/c3vd-v2/v4
'''
