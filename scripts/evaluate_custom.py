import argparse
import json
import os
import os.path as osp
from glob import glob
from itertools import product

import cv2
import imageio.v3 as iio
import numpy as np
import roma
import torch
from tqdm import tqdm

from flow3d.data.casual_dataset import load_cameras
from flow3d.data.utils import parse_tapir_track_info
from flow3d.metrics import mLPIPS, mPSNR, mSSIM
from flow3d.transforms import rt_to_mat4, solve_procrustes


def list_frames(directory: str, exts: tuple[str, ...]) -> list[str]:
    if not osp.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    normalized_exts = {
        ext if ext.startswith(".") else f".{ext}"
        for ext in (e.lower() for e in exts)
    }
    frames = []
    for path in sorted(glob(osp.join(directory, "*"))):
        if not osp.isfile(path):
            continue
        name, ext = osp.splitext(osp.basename(path))
        if ext.lower() in normalized_exts:
            frames.append(name)
    if not frames:
        raise RuntimeError(f"No frames with extensions {sorted(normalized_exts)} in {directory}")
    return frames


def load_image_stack(image_dir: str, frame_names: list[str]) -> np.ndarray:
    images = []
    for name in tqdm(frame_names, desc="Loading images"):
        path = osp.join(image_dir, f"{name}.png")
        if not osp.isfile(path):
            cand = glob(osp.join(image_dir, f"{name}.*"))
            if not cand:
                raise FileNotFoundError(f"Image for frame {name} not found in {image_dir}")
            path = cand[0]
        images.append(iio.imread(path))
    return np.asarray(images)


def load_depth_stack(depth_dir: str, frame_names: list[str]) -> np.ndarray:
    depths = []
    for name in tqdm(frame_names, desc="Loading depths"):
        path = osp.join(depth_dir, f"{name}.npy")
        if not osp.isfile(path):
            cand = glob(osp.join(depth_dir, f"{name}.npz"))
            if not cand:
                raise FileNotFoundError(f"Depth for frame {name} not found in {depth_dir}")
            path = cand[0]
            depth = np.load(path)
            if isinstance(depth, np.lib.npyio.NpzFile):
                keys = list(depth.files)
                if not keys:
                    raise ValueError(f"Empty depth npz: {path}")
                depth = depth[keys[0]]
            else:
                depth = depth
        else:
            depth = np.load(path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depths.append(depth.astype(np.float32))
    return np.asarray(depths)


def build_covisible_masks(mask_dir: str | None, imgs: np.ndarray) -> np.ndarray:
    if mask_dir is None or not osp.isdir(mask_dir):
        return np.ones(imgs.shape[:-1], dtype=np.uint8) * 255
    masks = []
    for frame_name in tqdm(sorted(list_frames(mask_dir, ("png", "jpg", "jpeg"))), desc="Loading covisibles"):
        path = osp.join(mask_dir, f"{frame_name}.png")
        if not osp.isfile(path):
            cand = glob(osp.join(mask_dir, f"{frame_name}.*"))
            if not cand:
                continue
            path = cand[0]
        mask = iio.imread(path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        masks.append(mask)
    if len(masks) != len(imgs):
        return np.ones(imgs.shape[:-1], dtype=np.uint8) * 255
    masks = np.asarray(masks)
    if masks.max() <= 1:
        masks = (masks > 0).astype(np.uint8) * 255
    return masks.astype(np.uint8)


def load_keypoints_from_tracks(
    track_dir: str,
    frame_names: list[str],
    key_indices: list[int],
    max_keypoints: int | None = None,
) -> np.ndarray:
    keypoints = []
    for idx in key_indices:
        frame = frame_names[idx]
        path = osp.join(track_dir, f"{frame}_{frame}.npy")
        if not osp.isfile(path):
            raise FileNotFoundError(f"Track file {path} not found")
        arr = np.load(path).astype(np.float32)
        coords = arr[:, :2]
        occlusions = torch.from_numpy(arr[:, 2])
        expected = torch.from_numpy(arr[:, 3])
        visibles, _, _ = parse_tapir_track_info(occlusions, expected)
        vis = visibles.cpu().numpy().astype(np.float32)
        if max_keypoints is not None and coords.shape[0] > max_keypoints:
            step = max(1, coords.shape[0] // max_keypoints)
            sel = slice(None, None, step)
            coords = coords[sel]
            vis = vis[sel]
        keypoints.append(np.concatenate([coords, vis[:, None]], axis=1))
    return np.asarray(keypoints)


def compute_keypoints_3d(
    keypoints_2d: np.ndarray,
    key_indices: list[int],
    train_depths: np.ndarray,
    train_Ks: np.ndarray,
    train_w2cs: np.ndarray,
) -> np.ndarray:
    keypoints_3d = []
    inv_w2cs = np.linalg.inv(train_w2cs)
    for frame_idx, keypoints in zip(key_indices, keypoints_2d):
        depth = train_depths[frame_idx]
        K = train_Ks[frame_idx]
        w2c_inv = inv_w2cs[frame_idx]
        visible = keypoints[:, 2] > 0.5
        coords = keypoints[:, :2].astype(np.float32)
        coords_map = coords[None]
        kp_depths = cv2.remap(
            depth.astype(np.float32),
            coords_map,
            None,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )[0]
        depth_valid = cv2.remap(
            (depth > 0).astype(np.float32),
            coords_map,
            None,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )[0]
        cam_coords = (
            np.einsum(
                "ij,pj->pi",
                np.linalg.inv(K),
                np.pad(coords, ((0, 0), (0, 1)), constant_values=1),
            )
            * kp_depths[:, None]
        )
        world_h = np.einsum(
            "ij,pj->pi",
            w2c_inv[:3],
            np.pad(cam_coords, ((0, 0), (0, 1)), constant_values=1),
        )
        vis_mask = (visible & (depth_valid > 0.5)).astype(np.float32)
        keypoints_3d.append(np.concatenate([world_h, vis_mask[:, None]], axis=1))
    return np.asarray(keypoints_3d)


def _load_pose_txt_cameras(
    pose_path: str,
    intrinsics_path: str,
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not osp.isfile(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    poses = np.loadtxt(pose_path)
    if poses.ndim == 1:
        poses = poses[None, :]
    if poses.shape[-1] != 16:
        raise ValueError(
            f"Expected flattened 4x4 matrices in pose file, got shape {poses.shape}"
        )
    c2ws = poses.reshape(-1, 4, 4).transpose(0, 2, 1)
    w2cs = np.linalg.inv(c2ws)

    if not osp.isfile(intrinsics_path):
        raise FileNotFoundError(
            f"Intrinsics file required for pose.txt cameras: {intrinsics_path}"
        )
    with open(intrinsics_path, "r") as f:
        intr_data = json.load(f)
    intr_dict = intr_data.get("rectified_pinhole") or intr_data.get(
        "original_pinhole_k1k2"
    )
    if intr_dict is None:
        raise ValueError(
            f"Intrinsics JSON must contain 'rectified_pinhole' or 'original_pinhole_k1k2': {intrinsics_path}"
        )

    src_w = float(intr_dict["width"])
    src_h = float(intr_dict["height"])
    sx = W / src_w if src_w > 0 else 1.0
    sy = H / src_h if src_h > 0 else 1.0
    fx = float(intr_dict["fx"]) * sx
    fy = float(intr_dict["fy"]) * sy
    cx = float(intr_dict["cx"]) * sx
    cy = float(intr_dict["cy"]) * sy
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    Ks = np.tile(K[None, ...], (len(w2cs), 1, 1))
    tstamps = np.arange(len(w2cs), dtype=np.int64)

    return (
        torch.from_numpy(w2cs).float(),
        torch.from_numpy(Ks).float(),
        torch.from_numpy(tstamps),
    )


def load_data_dict(
    data_dir: str,
    image_subdir: str,
    depth_subdir: str,
    track_subdir: str,
    camera_file: str,
    mask_subdir: str | None,
    max_keypoints: int | None,
    intrinsics_file: str | None,
) -> dict:
    image_dir = osp.join(data_dir, image_subdir)
    depth_dir = osp.join(data_dir, depth_subdir)
    track_dir = osp.join(data_dir, track_subdir)
    mask_dir = osp.join(data_dir, mask_subdir) if mask_subdir else None

    frame_names = list_frames(image_dir, ("png", "jpg", "jpeg"))
    val_imgs = load_image_stack(image_dir, frame_names)
    val_covisibles = build_covisible_masks(mask_dir, val_imgs)
    train_depths = load_depth_stack(depth_dir, frame_names)

    H, W = val_imgs.shape[1:3]
    camera_path = osp.join(data_dir, camera_file)
    if camera_file.lower().endswith(".txt"):
        intr_path = osp.join(data_dir, intrinsics_file or "intrinsics.json")
        w2cs, Ks, keyframe_tstamps = _load_pose_txt_cameras(camera_path, intr_path, H, W)
    else:
        w2cs, Ks, keyframe_tstamps = load_cameras(camera_path, H, W)
    train_Ks = Ks.numpy()
    train_w2cs = w2cs.numpy()

    key_indices = keyframe_tstamps.int().tolist()
    key_indices = [idx for idx in key_indices if 0 <= idx < len(frame_names)]
    if not key_indices:
        raise RuntimeError("No valid keyframe indices found in camera file.")

    key_indices_arr = np.asarray(key_indices, dtype=np.int32)

    keypoints_2d = load_keypoints_from_tracks(
        track_dir, frame_names, key_indices, max_keypoints
    )
    keypoints_3d = compute_keypoints_3d(
        keypoints_2d,
        key_indices,
        train_depths,
        train_Ks,
        train_w2cs,
    )

    time_ids = key_indices_arr
    time_pairs = np.asarray(list(product(time_ids, repeat=2)), dtype=np.int32)
    index_pairs = np.asarray(
        list(product(range(len(time_ids)), repeat=2)), dtype=np.int32
    )
    time_id_to_index = {int(tid): idx for idx, tid in enumerate(time_ids.tolist())}

    return {
        "frame_names": frame_names,
        "val_imgs": val_imgs,
        "val_covisibles": val_covisibles,
        "train_depths": train_depths,
        "train_Ks": train_Ks,
        "train_w2cs": train_w2cs,
        "keypoints_2d": keypoints_2d,
        "keypoints_3d": keypoints_3d,
        "time_ids": time_ids,
        "orig_time_ids": time_ids.copy(),
        "time_pairs": time_pairs,
        "index_pairs": index_pairs,
        "time_id_to_index": time_id_to_index,
    }


def load_result_dict(result_dir: str, val_names: list[str]) -> dict:
    rgb_dir = osp.join(result_dir, "rgb")
    if not osp.isdir(rgb_dir):
        rgb_dir = result_dir

    try:
        pred_val_imgs = np.asarray(
            [iio.imread(osp.join(rgb_dir, f"{name}.png")) for name in val_names]
        )
    except Exception:
        pred_val_imgs = None

    try:
        keypoints_dict = np.load(
            osp.join(result_dir, "keypoints.npz"), allow_pickle=True
        )
        if len(keypoints_dict) == 1 and "arr_0" in keypoints_dict:
            keypoints_dict = keypoints_dict["arr_0"].item()
        pred_keypoint_Ks = keypoints_dict.get("Ks")
        pred_keypoint_w2cs = keypoints_dict.get("w2cs")
        pred_keypoints_3d = keypoints_dict.get("pred_keypoints_3d")
        pred_train_depths = keypoints_dict.get("pred_train_depths")
        pred_visibilities = keypoints_dict.get("visibilities")
        pred_time_ids = keypoints_dict.get("time_ids")
        pred_orig_time_ids = keypoints_dict.get("orig_time_ids")
    except Exception:
        pred_keypoint_Ks = None
        pred_keypoint_w2cs = None
        pred_keypoints_3d = None
        pred_train_depths = None
        pred_visibilities = None
        pred_time_ids = None
        pred_orig_time_ids = None

    return {
        "pred_val_imgs": pred_val_imgs,
        "pred_train_depths": pred_train_depths,
        "pred_keypoint_Ks": pred_keypoint_Ks,
        "pred_keypoint_w2cs": pred_keypoint_w2cs,
        "pred_keypoints_3d": pred_keypoints_3d,
        "pred_visibilities": pred_visibilities,
        "pred_time_ids": pred_time_ids,
        "pred_orig_time_ids": pred_orig_time_ids,
    }


def align_tracking_data(data_dict: dict, result_dict: dict) -> tuple[dict, dict]:
    pred_orig_time_ids = result_dict.get("pred_orig_time_ids")
    if pred_orig_time_ids is None:
        pred_orig_time_ids = result_dict.get("pred_time_ids")
    if pred_orig_time_ids is None:
        return data_dict, result_dict

    pred_orig_time_ids = np.asarray(pred_orig_time_ids).astype(np.int32)
    time_id_to_index = data_dict.get("time_id_to_index")
    if time_id_to_index is None:
        return data_dict, result_dict

    data_indices = []
    pred_indices = []
    for i, tid in enumerate(pred_orig_time_ids.tolist()):
        if int(tid) in time_id_to_index:
            data_indices.append(time_id_to_index[int(tid)])
            pred_indices.append(i)

    if not data_indices:
        return data_dict, result_dict

    data_indices = np.asarray(data_indices, dtype=np.int32)
    pred_indices = np.asarray(pred_indices, dtype=np.int32)
    orig_time_ids = np.asarray(
        data_dict.get("orig_time_ids", data_dict["time_ids"])
    ).astype(np.int32)
    common_orig_time_ids = orig_time_ids[data_indices]
    num_frames = len(common_orig_time_ids)

    aligned_data = data_dict.copy()
    aligned_data["time_ids"] = np.arange(num_frames, dtype=np.int32)
    aligned_data["orig_time_ids"] = common_orig_time_ids
    aligned_data["keypoints_2d"] = data_dict["keypoints_2d"][data_indices].copy()
    aligned_data["keypoints_3d"] = data_dict["keypoints_3d"][data_indices].copy()
    aligned_data["train_Ks"] = data_dict["train_Ks"][common_orig_time_ids].copy()
    aligned_data["train_w2cs"] = data_dict["train_w2cs"][common_orig_time_ids].copy()
    aligned_data["train_depths"] = data_dict["train_depths"][common_orig_time_ids].copy()
    aligned_data["time_pairs"] = np.asarray(
        list(product(range(num_frames), repeat=2)), dtype=np.int32
    )
    aligned_data["index_pairs"] = np.asarray(
        list(product(range(num_frames), repeat=2)), dtype=np.int32
    )
    aligned_data["time_id_to_index"] = {
        int(tid): idx for idx, tid in enumerate(common_orig_time_ids.tolist())
    }

    aligned_result = result_dict.copy()
    if aligned_result.get("pred_keypoint_Ks") is not None:
        aligned_result["pred_keypoint_Ks"] = np.asarray(
            aligned_result["pred_keypoint_Ks"]
        )[pred_indices]
    if aligned_result.get("pred_keypoint_w2cs") is not None:
        aligned_result["pred_keypoint_w2cs"] = np.asarray(
            aligned_result["pred_keypoint_w2cs"]
        )[pred_indices]
    if aligned_result.get("pred_train_depths") is not None:
        aligned_result["pred_train_depths"] = np.asarray(
            aligned_result["pred_train_depths"]
        )[pred_indices]
    if aligned_result.get("pred_time_ids") is not None:
        aligned_result["pred_time_ids"] = np.asarray(
            aligned_result["pred_time_ids"], dtype=np.int32
        )[pred_indices]
    if aligned_result.get("pred_orig_time_ids") is not None:
        aligned_result["pred_orig_time_ids"] = np.asarray(
            aligned_result["pred_orig_time_ids"], dtype=np.int32
        )[pred_indices]

    return aligned_data, aligned_result


def evaluate_3d_tracking(data_dict: dict, result_dict: dict) -> tuple[float, float, float]:
    train_Ks = data_dict["train_Ks"]
    train_w2cs = data_dict["train_w2cs"]
    keypoints_3d = data_dict["keypoints_3d"]
    time_ids = data_dict["time_ids"]
    time_pairs = data_dict["time_pairs"]
    index_pairs = data_dict["index_pairs"]
    orig_time_ids = data_dict.get("orig_time_ids", time_ids)

    pred_keypoint_Ks = result_dict["pred_keypoint_Ks"]
    pred_keypoint_w2cs = result_dict["pred_keypoint_w2cs"]
    pred_keypoints_3d = result_dict["pred_keypoints_3d"]

    if pred_keypoint_Ks is None or pred_keypoint_w2cs is None or pred_keypoints_3d is None:
        raise RuntimeError("Predicted keypoint data not found in results.")

    if not np.allclose(train_Ks, pred_keypoint_Ks):
        print("Inconsistent camera intrinsics between GT and predictions.")

    keypoint_w2cs = train_w2cs
    q, t, s = solve_procrustes(
        torch.from_numpy(np.linalg.inv(pred_keypoint_w2cs)[:, :3, -1]).to(torch.float32),
        torch.from_numpy(np.linalg.inv(keypoint_w2cs)[:, :3, -1]).to(torch.float32),
    )[0]
    R = roma.unitquat_to_rotmat(q.roll(-1, dims=-1))
    pred_keypoints_3d = np.einsum(
        "ij,...j->...i",
        rt_to_mat4(R, t, s).numpy().astype(np.float64),
        np.pad(pred_keypoints_3d, ((0, 0), (0, 0), (0, 1)), constant_values=1),
    )
    pred_keypoints_3d = pred_keypoints_3d[..., :3] / pred_keypoints_3d[..., 3:]

    pair_keypoints_3d = keypoints_3d[index_pairs]
    is_covisible = (pair_keypoints_3d[:, :, :, -1] == 1).all(axis=1)
    target_keypoints_3d = pair_keypoints_3d[:, 1, :, :3]

    epes = []
    for i in range(len(time_pairs)):
        epes.append(
            np.linalg.norm(
                target_keypoints_3d[i][is_covisible[i]]
                - pred_keypoints_3d[i][is_covisible[i]],
                axis=-1,
            )
        )

    epe = np.mean([frame_epes.mean() for frame_epes in epes if len(frame_epes) > 0]).item()
    pck_3d_10cm = np.mean(
        [(frame_epes < 0.1).mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()
    pck_3d_5cm = np.mean(
        [(frame_epes < 0.05).mean() for frame_epes in epes if len(frame_epes) > 0]
    ).item()

    print(f"3D tracking EPE: {epe:.4f}")
    print(f"3D tracking PCK (10cm): {pck_3d_10cm:.4f}")
    print(f"3D tracking PCK (5cm): {pck_3d_5cm:.4f}")
    print("-----------------------------")
    return epe, pck_3d_10cm, pck_3d_5cm


def project(Ks, w2cs, pts):
    N = Ks.shape[0]
    pts = pts.swapaxes(0, 1).reshape(N, -1, 3)

    pts_homogeneous = np.concatenate([pts, np.ones_like(pts[..., -1:])], axis=-1)
    pts_homogeneous = np.matmul(w2cs[:, :3], pts_homogeneous.swapaxes(1, 2)).swapaxes(1, 2)
    projected_pts = np.matmul(Ks, pts_homogeneous.swapaxes(1, 2)).swapaxes(1, 2)

    depths = projected_pts[..., 2:3]
    projected_pts = projected_pts[..., :2] / np.clip(depths, a_min=1e-6, a_max=None)
    projected_pts = projected_pts.reshape(N, N, -1, 2).swapaxes(0, 1)
    depths = depths.reshape(N, N, -1).swapaxes(0, 1)
    return projected_pts, depths


def evaluate_2d_tracking(data_dict: dict, result_dict: dict) -> tuple[float, float, float]:
    train_w2cs = data_dict["train_w2cs"]
    keypoints_2d = data_dict["keypoints_2d"]
    visibilities = keypoints_2d[..., -1].astype(np.bool_)
    time_ids = data_dict["time_ids"]
    num_frames = len(time_ids)
    num_pts = keypoints_2d.shape[1]

    pred_train_depths = result_dict["pred_train_depths"]
    pred_keypoint_Ks = result_dict["pred_keypoint_Ks"]
    pred_keypoint_w2cs = result_dict["pred_keypoint_w2cs"]
    pred_keypoints_3d = result_dict["pred_keypoints_3d"]
    if pred_keypoint_Ks is None or pred_keypoint_w2cs is None or pred_keypoints_3d is None:
        raise RuntimeError("Predicted keypoint data not found for 2D tracking evaluation.")

    pred_keypoints_3d = pred_keypoints_3d.reshape(num_frames, -1, num_pts, 3)
    keypoint_w2cs = train_w2cs[time_ids]
    s = solve_procrustes(
        torch.from_numpy(np.linalg.inv(pred_keypoint_w2cs)[:, :3, -1]).to(torch.float32),
        torch.from_numpy(np.linalg.inv(keypoint_w2cs)[:, :3, -1]).to(torch.float32),
    )[0][-1].item()

    target_points = keypoints_2d[None].repeat(num_frames, axis=0)[..., :2]
    target_visibilities = visibilities[None].repeat(num_frames, axis=0)

    pred_points, pred_depths = project(
        pred_keypoint_Ks, pred_keypoint_w2cs, pred_keypoints_3d
    )

    if result_dict["pred_visibilities"] is not None:
        pred_visibilities = result_dict["pred_visibilities"].reshape(
            num_frames, -1, num_pts
        )
    else:
        rendered_depths = []
        for i, points in zip(
            data_dict["index_pairs"][:, -1],
            pred_points.reshape(-1, pred_points.shape[2], 2),
        ):
            rendered_depths.append(
                cv2.remap(
                    pred_train_depths[i].astype(np.float32),
                    points[None].astype(np.float32),
                    None,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )[0]
            )
        rendered_depths = np.array(rendered_depths).reshape(num_frames, -1, num_pts)
        pred_visibilities = (np.abs(rendered_depths - pred_depths) * s) < 0.05

    one_hot_eye = np.eye(target_points.shape[0])[..., None].repeat(num_pts, axis=-1)
    evaluation_points = one_hot_eye == 0
    for i in range(num_frames):
        evaluation_points[i, :, ~visibilities[i]] = False

    occ_acc = np.sum(
        np.equal(pred_visibilities, target_visibilities) & evaluation_points
    ) / np.sum(evaluation_points)

    all_frac_within = []
    all_jaccard = []
    for thresh in [4, 8, 16, 32, 64]:
        within_dist = np.sum(
            np.square(pred_points - target_points),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, target_visibilities)
        count_correct = np.sum(is_correct & evaluation_points)
        count_visible_points = np.sum(target_visibilities & evaluation_points)
        frac_correct = count_visible_points and count_correct / count_visible_points or 0.0
        all_frac_within.append(frac_correct)

        true_positives = np.sum(is_correct & pred_visibilities & evaluation_points)
        gt_positives = np.sum(target_visibilities & evaluation_points)
        false_positives = (~target_visibilities) & pred_visibilities
        false_positives = false_positives | ((~within_dist) & pred_visibilities)
        false_positives = np.sum(false_positives & evaluation_points)
        jaccard = true_positives / (gt_positives + false_positives)
        all_jaccard.append(jaccard)

    AJ = np.mean(all_jaccard)
    APCK = np.mean(all_frac_within)

    print(f"2D tracking AJ: {AJ:.4f}")
    print(f"2D tracking avg PCK: {APCK:.4f}")
    print(f"2D tracking occlusion accuracy: {occ_acc:.4f}")
    print("-----------------------------")
    return AJ, APCK, occ_acc


def evaluate_nv(data_dict: dict, result_dict: dict) -> tuple[float, float, float]:
    val_imgs = torch.from_numpy(data_dict["val_imgs"])[..., :3]
    val_covisibles = torch.from_numpy(data_dict["val_covisibles"]).float()
    pred_val_imgs_np = result_dict["pred_val_imgs"]
    if pred_val_imgs_np is None:
        raise RuntimeError("No predicted RGB frames found.")
    pred_val_imgs = torch.from_numpy(pred_val_imgs_np)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    psnr_metric = mPSNR().to(device)
    ssim_metric = mSSIM().to(device)
    lpips_metric = mLPIPS().to(device)

    for i in range(len(val_imgs)):
        val_img = (val_imgs[i] / 255.0).to(device)
        pred_val_img = (pred_val_imgs[i] / 255.0).to(device)
        val_covisible = (val_covisibles[i] / 255.0).to(device)
        psnr_metric.update(pred_val_img, val_img, val_covisible)
        ssim_metric.update(
            pred_val_img[None], val_img[None], val_covisible[None]
        )
        lpips_metric.update(
            pred_val_img[None], val_img[None], val_covisible[None]
        )

    mpsnr = psnr_metric.compute().item()
    mssim = ssim_metric.compute().item()
    mlpips = lpips_metric.compute().item()
    print(f"NV mPSNR: {mpsnr:.4f}")
    print(f"NV mSSIM: {mssim:.4f}")
    print(f"NV mLPIPS: {mlpips:.4f}")
    return mpsnr, mssim, mlpips


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate custom Shape-of-Motion results (PSNR/SSIM/LPIPS + tracking)."
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument(
        "--seq_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of sequence names under data_dir/result_dir.",
    )
    parser.add_argument("--image_subdir", type=str, default="images")
    parser.add_argument("--depth_subdir", type=str, default="aligned_depth_anything")
    parser.add_argument("--tracks_subdir", type=str, default="bootstapir")
    parser.add_argument("--camera_file", type=str, default="droid_recon.npy")
    parser.add_argument(
        "--intrinsics_file",
        type=str,
        default="intrinsics.json",
        help="Intrinsics JSON file (used when camera_file is pose.txt).",
    )
    parser.add_argument("--mask_subdir", type=str, default=None)
    parser.add_argument("--max_keypoints", type=int, default=None)

    args = parser.parse_args()

    seq_names = args.seq_names or [""]

    epe_all, pck_3d_10cm_all, pck_3d_5cm_all = [], [], []
    AJ_all, APCK_all, occ_acc_all = [], [], []
    mpsnr_all, mssim_all, mlpips_all = [], [], []

    for seq_name in seq_names:
        data_dir = osp.join(args.data_dir, seq_name) if seq_name else args.data_dir
        result_dir = osp.join(args.result_dir, seq_name, "results") if seq_name else osp.join(args.result_dir, "results")
        if not osp.isdir(result_dir):
            result_dir = osp.join(args.result_dir, seq_name) if seq_name else args.result_dir
        if not osp.isdir(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} not found")
        if not osp.isdir(result_dir):
            raise FileNotFoundError(f"Result directory {result_dir} not found")

        print("=========================================")
        print(f"Evaluating {seq_name or osp.basename(data_dir)}")
        print("=========================================")

        max_keypoints = args.max_keypoints
        if max_keypoints is None:
            path_parts = osp.normpath(data_dir).split(os.sep)
            if any("c3vd" in part.lower() for part in path_parts):
                max_keypoints = 64

        data_dict = load_data_dict(
            data_dir,
            args.image_subdir,
            args.depth_subdir,
            args.tracks_subdir,
            args.camera_file,
            args.mask_subdir,
            max_keypoints,
            args.intrinsics_file,
        )
        result_dict = load_result_dict(result_dir, data_dict["frame_names"])
        data_dict, result_dict = align_tracking_data(data_dict, result_dict)

        if result_dict["pred_keypoints_3d"] is not None:
            epe, pck10, pck5 = evaluate_3d_tracking(data_dict, result_dict)
            AJ, APCK, occ_acc = evaluate_2d_tracking(data_dict, result_dict)
            epe_all.append(epe)
            pck_3d_10cm_all.append(pck10)
            pck_3d_5cm_all.append(pck5)
            AJ_all.append(AJ)
            APCK_all.append(APCK)
            occ_acc_all.append(occ_acc)
        else:
            print("Keypoint predictions not found; skipping tracking metrics.")

        if result_dict["pred_val_imgs"] is not None and len(data_dict["val_imgs"]) > 0:
            mpsnr, mssim, mlpips = evaluate_nv(data_dict, result_dict)
            mpsnr_all.append(mpsnr)
            mssim_all.append(mssim)
            mlpips_all.append(mlpips)
        else:
            print("No NV results found.")

    if epe_all:
        print(f"mean 3D tracking EPE: {np.mean(epe_all):.4f}")
        print(f"mean 3D tracking PCK (10cm): {np.mean(pck_3d_10cm_all):.4f}")
        print(f"mean 3D tracking PCK (5cm): {np.mean(pck_3d_5cm_all):.4f}")
    if AJ_all:
        print(f"mean 2D tracking AJ: {np.mean(AJ_all):.4f}")
        print(f"mean 2D tracking avg PCK: {np.mean(APCK_all):.4f}")
        print(f"mean 2D tracking occlusion accuracy: {np.mean(occ_acc_all):.4f}")
    if mpsnr_all:
        print(f"mean NV mPSNR: {np.mean(mpsnr_all):.4f}")
        print(f"mean NV mSSIM: {np.mean(mssim_all):.4f}")
        print(f"mean NV mLPIPS: {np.mean(mlpips_all):.4f}")


if __name__ == "__main__":
    main()
