import json
import os
import os.path as osp
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import Literal, cast

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from loguru import logger as guru
from roma import roma
from torch.utils.data import Dataset

from flow3d.data.base_dataset import BaseDataset
from flow3d.data.utils import (
    SceneNormDict,
    get_tracks_3d_for_query_frame,
    median_filter_2d,
    normalize_coords,
    parse_tapir_track_info,
)
from flow3d.transforms import rt_to_mat4



@dataclass
class C3VDDataConfig:
    data_dir: str
    start: int = 0
    end: int = -1
    max_frames: int | None = None
    val_stride: int = 8
    res: str = ""
    image_type: str = "images"
    mask_type: str = "masks"
    depth_type: str = "depths"
    track_2d_type: Literal["bootstapir", "tapir", "cotracker"] = "bootstapir"
    camera_type: Literal["pose_txt", "droid_recon", "megasam"] = "pose_txt"
    pose_name: str = "pose.txt"
    intrinsics_name: str = "intrinsics.json"
    depth_format: Literal["depth", "disparity"] = "depth"
    mask_erosion_radius: int = 0
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False


def _resolve_subdir(root: str, subdir: str, res: str) -> str:
    if res:
        return osp.join(root, subdir, res)
    return osp.join(root, subdir)


class C3VDDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        start: int = 0,
        end: int = -1,
        max_frames: int | None = None,
        res: str = "",
        image_type: str = "images",
        mask_type: str = "masks",
        depth_type: str = "depths",
        track_2d_type: Literal["bootstapir", "tapir", "cotracker"] = "bootstapir",
        camera_type: str = "pose_txt",
        pose_name: str = "pose.txt",
        intrinsics_name: str = "intrinsics.json",
        depth_format: Literal["depth", "disparity"] = "depth",
        mask_erosion_radius: int = 0,
        scene_norm_dict: SceneNormDict | None = None,
        num_targets_per_frame: int = 4,
        load_from_cache: bool = False,
        frame_stride: int = 1,
        split: Literal["train", "val"] = "train",
        **_,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.res = res
        self.depth_type = depth_type
        self.depth_format = depth_format
        self.num_targets_per_frame = num_targets_per_frame
        self.load_from_cache = load_from_cache
        self.has_validation = False
        self.mask_erosion_radius = mask_erosion_radius
        self.camera_type = camera_type
        self._max_frames = max_frames

        self.pose_path = Path(data_dir) / pose_name
        self.intrinsics_path = Path(data_dir) / intrinsics_name

        self.img_dir = _resolve_subdir(data_dir, image_type, res)
        img_files = sorted(os.listdir(self.img_dir))
        if not img_files:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
        self.img_ext = os.path.splitext(img_files[0])[1]
        self.depth_dir = _resolve_subdir(data_dir, depth_type, res)
        if not osp.isdir(self.depth_dir):
            raise FileNotFoundError(f"Depth directory not found: {self.depth_dir}")
        self.mask_dir = _resolve_subdir(data_dir, mask_type, res)
        if not osp.isdir(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        tracks_root = track_2d_type.strip("/")
        self.tracks_dir = _resolve_subdir(data_dir, tracks_root, res)
        if tracks_root and not osp.isdir(self.tracks_dir):
            raise FileNotFoundError(f"Track directory not found: {self.tracks_dir}")
        self.cache_dir = _resolve_subdir(data_dir, "flow3d_preprocessed", res)
        self.split = split
        self.frame_stride = max(1, frame_stride)
        self.track_2d_type = track_2d_type

        sample_img = imageio.imread(osp.join(self.img_dir, img_files[0]))
        self._img_hw = sample_img.shape[:2]

        frame_names_full = [os.path.splitext(p)[0] for p in img_files]
        full_name_to_idx = {name: idx for idx, name in enumerate(frame_names_full)}

        splits_dir = Path(data_dir) / "splits"
        split_pairs: list[tuple[str, int]] | None = None
        self.has_validation = False
        if splits_dir.is_dir():
            val_path = splits_dir / "val.json"
            if val_path.exists():
                with val_path.open("r") as f:
                    val_dict = json.load(f)
                self.has_validation = len(val_dict.get("frame_names", [])) > 0
            split_path = splits_dir / f"{split}.json"
            if split_path.exists():
                with split_path.open("r") as f:
                    split_dict = json.load(f)
                split_names = split_dict.get("frame_names", [])
                split_time_ids = split_dict.get("time_ids", [])
                split_pairs = []
                for idx_in_list, name in enumerate(split_names):
                    orig_idx: int | None = None
                    if idx_in_list < len(split_time_ids):
                        try:
                            orig_idx = int(split_time_ids[idx_in_list])
                        except (TypeError, ValueError):
                            orig_idx = None
                    if orig_idx is None or orig_idx < 0 or orig_idx >= len(frame_names_full):
                        orig_idx = full_name_to_idx.get(name)
                    if orig_idx is None:
                        guru.warning(
                            f"Skipping frame {name} listed in {split_path} but missing from images"
                        )
                        continue
                    split_pairs.append((name, orig_idx))
                if not split_pairs:
                    split_pairs = None

        total_available = len(split_pairs) if split_pairs is not None else len(frame_names_full)
        if end == -1 or end > total_available:
            end = total_available
        if self._max_frames is not None:
            end = min(end, start + self._max_frames)
        end = max(start, min(end, total_available))

        base_indices = list(range(start, end))
        selected_indices = base_indices[:: self.frame_stride]
        if not selected_indices:
            raise ValueError(
                f"No frames selected for dataset with frame_stride={self.frame_stride}."
            )

        if split_pairs is not None:
            frame_entries = [split_pairs[i] for i in selected_indices]
        else:
            frame_entries = [(frame_names_full[i], i) for i in selected_indices]

        self.start = start
        self.end = end
        self.frame_idcs = torch.tensor([idx for _, idx in frame_entries], dtype=torch.long)
        self.frame_names = [name for name, _ in frame_entries]
        self._frame_name_to_idx = {name: i for i, name in enumerate(self.frame_names)}

        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.depths: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.masks: list[torch.Tensor | None] = [None for _ in self.frame_names]

        # load cameras
        if camera_type == "pose_txt":
            if not self.pose_path.exists():
                raise FileNotFoundError(f"Missing pose file: {self.pose_path}")
            poses = np.loadtxt(self.pose_path)
            if poses.ndim == 1:
                poses = poses[None, :]
            if poses.shape[-1] != 16:
                raise ValueError(
                    f"Expected pose.txt rows of length 16, got shape {poses.shape}"
                )
            c2ws = poses.reshape(-1, 4, 4)
            c2ws = c2ws.transpose(0, 2, 1)
            if c2ws.shape[0] < len(frame_names_full):
                raise ValueError(
                    f"pose.txt has {c2ws.shape[0]} entries, expected >= {len(frame_names_full)}"
                )
            c2ws = c2ws[: len(frame_names_full)]
            w2cs = np.linalg.inv(c2ws)

            if not self.intrinsics_path.exists():
                raise FileNotFoundError(
                    f"Missing intrinsics file: {self.intrinsics_path}"
                )
            with open(self.intrinsics_path, "r") as f:
                intr_data = json.load(f)
            intr_dict = (
                intr_data.get("rectified_pinhole")
                or intr_data.get("original_pinhole_k1k2")
            )
            if intr_dict is None:
                raise ValueError(
                    f"Intrinsics JSON must contain 'rectified_pinhole' or 'original_pinhole_k1k2': {self.intrinsics_path}"
                )
            src_w = float(intr_dict["width"])
            src_h = float(intr_dict["height"])
            img_h, img_w = self._img_hw
            sx = img_w / src_w if src_w > 0 else 1.0
            sy = img_h / src_h if src_h > 0 else 1.0
            fx = float(intr_dict["fx"]) * sx
            fy = float(intr_dict["fy"]) * sy
            cx = float(intr_dict["cx"]) * sx
            cy = float(intr_dict["cy"]) * sy
            K = torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
            Ks = K.unsqueeze(0).repeat(len(frame_names_full), 1, 1)
            w2cs = torch.from_numpy(w2cs).float()
            tstamps = torch.arange(len(frame_names_full))
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")

        w2cs = w2cs[self.frame_idcs]
        Ks = Ks[self.frame_idcs]
        if not torch.is_tensor(tstamps):
            tstamps = torch.from_numpy(tstamps)
        tstamps = tstamps[self.frame_idcs]

        assert (
            len(self.frame_names) == len(w2cs) == len(Ks)
        ), f"{len(self.frame_names)}, {len(w2cs)}, {len(Ks)}"

        self.w2cs = w2cs
        self.Ks = Ks
        if not torch.is_tensor(tstamps):
            tstamps = torch.from_numpy(tstamps)
        frame_idx_map = {int(idx): pos for pos, idx in enumerate(self.frame_idcs.tolist())}
        selected = [
            frame_idx_map[int(ts.item())]
            for ts in tstamps
            if int(ts.item()) in frame_idx_map
        ]
        if selected:
            self._keyframe_idcs = torch.tensor(selected, dtype=torch.long)
        else:
            self._keyframe_idcs = torch.arange(len(self.frame_names), dtype=torch.long)
        self.scale = 1

        stride = 32 # lowey keyframe 
        if stride > 1 and self._keyframe_idcs.numel() > stride:
            self._keyframe_idcs = self._keyframe_idcs[::stride]

        if scene_norm_dict is None:
            cached_scene_norm_dict_path = os.path.join(
                self.cache_dir, "scene_norm_dict.pth"
            )
            if os.path.exists(cached_scene_norm_dict_path) and self.load_from_cache:
                guru.info("loading cached scene norm dict...")
                scene_norm_dict = torch.load(
                    os.path.join(self.cache_dir, "scene_norm_dict.pth")
                )
            else:
                tracks_3d = self.get_tracks_3d(5000, step=self.num_frames // 10)[0]
                scale, transfm = compute_scene_norm(tracks_3d, self.w2cs)
                scene_norm_dict = SceneNormDict(scale=scale, transfm=transfm)
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save(scene_norm_dict, cached_scene_norm_dict_path)

        # transform cameras
        self.scene_norm_dict = cast(SceneNormDict, scene_norm_dict)
        self.scale = self.scene_norm_dict["scale"]
        transform = self.scene_norm_dict["transfm"]
        guru.info(f"scene norm {self.scale=}, {transform=}")
        self.w2cs = torch.einsum("nij,jk->nik", self.w2cs, torch.linalg.inv(transform))
        self.w2cs[:, :3, 3] /= self.scale

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    @property
    def keyframe_idcs(self) -> torch.Tensor:
        return self._keyframe_idcs

    def __len__(self):
        return len(self.frame_names)

    def get_w2cs(self) -> torch.Tensor:
        return self.w2cs

    def get_Ks(self) -> torch.Tensor:
        return self.Ks

    def get_img_wh(self) -> tuple[int, int]:
        return self.get_image(0).shape[1::-1]

    def get_image(self, index) -> torch.Tensor:
        if self.imgs[index] is None:
            self.imgs[index] = self.load_image(index)
        img = cast(torch.Tensor, self.imgs[index])
        return img

    def get_mask(self, index) -> torch.Tensor:
        if self.masks[index] is None:
            self.masks[index] = self.load_mask(index)
        mask = cast(torch.Tensor, self.masks[index])
        return mask

    def get_depth(self, index) -> torch.Tensor:
        if self.depths[index] is None:
            self.depths[index] = self.load_depth(index)
        return self.depths[index] / self.scale

    def load_image(self, index) -> torch.Tensor:
        path = f"{self.img_dir}/{self.frame_names[index]}{self.img_ext}"
        return torch.from_numpy(imageio.imread(path)).float() / 255.0

    def load_mask(self, index) -> torch.Tensor:
        path = f"{self.mask_dir}/{self.frame_names[index]}.png"
        r = self.mask_erosion_radius
        try:
            mask = imageio.imread(path)
        except:
            path = f"{self.mask_dir}/{self.frame_names[index]}.jpg"
            mask = imageio.imread(path)
        fg_mask = mask.reshape((*mask.shape[:2], -1)).max(axis=-1) > 0
        if r > 0:
            kernel = np.ones((r, r), np.uint8)
            fg_mask_erode = cv2.erode(
                fg_mask.astype(np.uint8), kernel, iterations=1
            )
        else:
            fg_mask_erode = fg_mask.astype(np.uint8)
        out_mask = np.ones_like(fg_mask, dtype=np.float32)
        return torch.from_numpy(out_mask).float()

    def load_depth(self, index) -> torch.Tensor:
        path = f"{self.depth_dir}/{self.frame_names[index]}.npy"
        if not osp.exists(path):
            raise FileNotFoundError(f"Missing depth file: {path}")
        depth_arr = np.load(path)
        depth = torch.from_numpy(depth_arr).float()
        return depth

    def load_target_tracks(
        self, query_index: int, target_indices: list[int], dim: int = 1
    ):
        """
        tracks are 2d, occs and uncertainties
        :param dim (int), default 1: dimension to stack the time axis
        return (N, T, 4) if dim=1, (T, N, 4) if dim=0
        """

        q_name = self.frame_names[query_index]
        all_tracks = []
        for ti in target_indices:
            t_name = self.frame_names[ti]
            path = f"{self.tracks_dir}/{q_name}_{t_name}.npy"
            tracks = np.load(path).astype(np.float32)
            all_tracks.append(tracks)
        return torch.from_numpy(np.stack(all_tracks, axis=dim))

    def get_tracks_3d(
        self, num_samples: int, start: int = 0, end: int = -1, step: int = 1, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_frames = self.num_frames
        if end < 0:
            end = num_frames + 1 + end
        query_idcs = list(range(start, end, step))
        target_idcs = list(range(start, end, step))
        masks = torch.stack([self.get_mask(i) for i in target_idcs], dim=0)
        fg_masks = (masks == 1).float()
        depths = torch.stack([self.get_depth(i) for i in target_idcs], dim=0)
        inv_Ks = torch.linalg.inv(self.Ks[target_idcs])
        c2ws = torch.linalg.inv(self.w2cs[target_idcs])

        num_per_query_frame = int(np.ceil(num_samples / len(query_idcs)))
        cur_num = 0

        tracks_all_queries = []
        for q_idx in query_idcs:
            # (N, T, 4)
            tracks_2d = self.load_target_tracks(q_idx, target_idcs)
            num_sel = int(
                min(num_per_query_frame, num_samples - cur_num, len(tracks_2d))
            )

            if num_sel < len(tracks_2d):
                num_sel = num_per_query_frame
                sel_idcs = np.random.choice(len(tracks_2d), num_sel, replace=False)
                tracks_2d = tracks_2d[sel_idcs]
            cur_num += tracks_2d.shape[0]
            img = self.get_image(q_idx)
            tidx = target_idcs.index(q_idx)
            tracks_tuple = get_tracks_3d_for_query_frame(
                tidx, img, tracks_2d, depths, fg_masks, inv_Ks, c2ws
            )
            tracks_all_queries.append(tracks_tuple)
        tracks_3d, colors, visibles, invisibles, confidences = map(
            partial(torch.cat, dim=0), zip(*tracks_all_queries)
        )

        return tracks_3d, visibles, invisibles, confidences, colors

    def get_bkgd_points(
        self,
        num_samples: int,
        use_kf_tstamps: bool = True,
        stride: int = 8,
        down_rate: int = 8,
        min_per_frame: int = 64,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        empty = torch.empty((0, 3), dtype=torch.float32)
        return empty, empty.clone(), empty.clone()

    def __getitem__(self, index: int):
        if self.split == "train":
            index = np.random.randint(0, self.num_frames)
        else:
            index = int(index) % self.num_frames
        data = {
            # ().
            "frame_names": self.frame_names[index],
            # ().
            "ts": torch.tensor(index),
            # (4, 4).
            "w2cs": self.w2cs[index],
            # (3, 3).
            "Ks": self.Ks[index],
            # (H, W, 3).
            "imgs": self.get_image(index),
            "depths": self.get_depth(index),
        }

        tri_mask = self.get_mask(index)
        valid_mask = tri_mask != 0  # not fg or bg
        mask = tri_mask == 1  # fg mask
        data["masks"] = mask.float()
        data["valid_masks"] = valid_mask.float()

        # (P, 2)
        query_tracks = self.load_target_tracks(index, [index])[:, 0, :2]
        target_inds = torch.from_numpy(
            np.random.choice(
                self.num_frames, (self.num_targets_per_frame,), replace=False
            )
        )
        # (N, P, 4)
        target_tracks = self.load_target_tracks(index, target_inds.tolist(), dim=0)
        data["query_tracks_2d"] = query_tracks
        data["target_ts"] = target_inds
        data["target_w2cs"] = self.w2cs[target_inds]
        data["target_Ks"] = self.Ks[target_inds]
        data["target_tracks_2d"] = target_tracks[..., :2]
        # (N, P).
        (
            data["target_visibles"],
            data["target_invisibles"],
            data["target_confidences"],
        ) = parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])
        # (N, H, W)
        target_depths = torch.stack([self.get_depth(i) for i in target_inds], dim=0)
        H, W = target_depths.shape[-2:]
        data["target_track_depths"] = F.grid_sample(
            target_depths[:, None],
            normalize_coords(target_tracks[..., None, :2], H, W),
            align_corners=True,
            padding_mode="border",
        )[:, 0, :, 0]
        return data


class C3VDDatasetKeypointView(Dataset):
    """Dataset view yielding frame pairs with Track-based keypoints."""

    def __init__(
        self,
        dataset: "C3VDDataset",
        max_keypoints: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        assert self.dataset.split == "train", "Keypoint view expects training split"
        self.max_keypoints = max_keypoints

        self.dataset_indices = self.dataset.keyframe_idcs.clone()
        self.original_time_ids = self.dataset.frame_idcs[self.dataset_indices].clone()
        self.time_ids = self.dataset_indices.clone()
        self.time_pairs = torch.tensor(
            list(product(self.dataset_indices.tolist(), repeat=2)), dtype=torch.long
        )
        self.index_pairs = torch.tensor(
            list(product(range(len(self.dataset_indices)), repeat=2)), dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.time_pairs)

    def _sample_tracks(
        self, query_idx: int, target_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tracks = self.dataset.load_target_tracks(
            query_idx, [query_idx, target_idx], dim=0
        ).float()  # (2, P, 4)
        coords = tracks[..., :2]
        visibles, _, confidences = parse_tapir_track_info(
            tracks[..., 2], tracks[..., 3]
        )
        vis_mask = visibles.float()
        if self.max_keypoints is not None and coords.shape[1] > self.max_keypoints:
            conf_scores = confidences[0]
            topk = torch.topk(conf_scores, self.max_keypoints).indices
            coords = coords[:, topk]
            vis_mask = vis_mask[:, topk]
        return coords, vis_mask

    def __getitem__(self, index: int) -> dict:
        ts = self.time_pairs[index]
        query_idx = int(ts[0].item())
        target_idx = int(ts[1].item())
        coords, vis_mask = self._sample_tracks(query_idx, target_idx)
        keypoints = torch.cat([coords, vis_mask.unsqueeze(-1)], dim=-1)

        imgs = torch.stack(
            [self.dataset.get_image(i) for i in (query_idx, target_idx)], dim=0
        )

        return {
            "ts": ts,
            "w2cs": self.dataset.w2cs[ts],
            "Ks": self.dataset.Ks[ts],
            "imgs": imgs,
            "keypoints": keypoints,
        }


class C3VDDatasetVideoView(Dataset):
    """Dataset view used to render training-sequence videos."""

    def __init__(self, dataset: "C3VDDataset") -> None:
        super().__init__()
        self.dataset = dataset
        assert self.dataset.split == "train", "Video view expects training split"
        # Default to 15 fps if the underlying dataset does not encode it.
        self.fps = getattr(dataset, "fps", 15.0)

    def __len__(self) -> int:
        return self.dataset.num_frames

    def __getitem__(self, index: int) -> dict:
        return {
            "frame_names": self.dataset.frame_names[index],
            "ts": index,
            "w2cs": self.dataset.w2cs[index],
            "Ks": self.dataset.Ks[index],
            "imgs": self.dataset.get_image(index),
            "depths": self.dataset.get_depth(index),
            "masks": self.dataset.get_mask(index),
        }


def load_cameras(
    path: str, H: int, W: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert os.path.exists(path), f"Camera file {path} does not exist."
    recon = np.load(path, allow_pickle=True).item()
    guru.debug(f"{recon.keys()=}")
    traj_c2w = recon["traj_c2w"]  # (N, 4, 4)
    h, w = recon["img_shape"]
    sy, sx = H / h, W / w
    traj_w2c = np.linalg.inv(traj_c2w)
    fx, fy, cx, cy = recon["intrinsics"]  # (4,)
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])  # (3, 3)
    Ks = np.tile(K[None, ...], (len(traj_c2w), 1, 1))  # (N, 3, 3)
    kf_tstamps = recon["tstamps"].astype("int")
    return (
        torch.from_numpy(traj_w2c).float(),
        torch.from_numpy(Ks).float(),
        torch.from_numpy(kf_tstamps),
    )


def compute_scene_norm(
    X: torch.Tensor, w2cs: torch.Tensor
) -> tuple[float, torch.Tensor]:
    """
    :param X: [N*T, 3]
    :param w2cs: [N, 4, 4]
    """
    X = X.reshape(-1, 3)
    scene_center = X.mean(dim=0)
    X = X - scene_center[None]
    min_scale = X.quantile(0.05, dim=0)
    max_scale = X.quantile(0.95, dim=0)
    scale = (max_scale - min_scale).max().item() / 2.0
    original_up = -F.normalize(w2cs[:, 1, :3].mean(0), dim=-1)
    target_up = original_up.new_tensor([0.0, 0.0, 1.0])
    R = roma.rotvec_to_rotmat(
        F.normalize(original_up.cross(target_up), dim=-1)
        * original_up.dot(target_up).acos_()
    )
    transfm = rt_to_mat4(R, torch.einsum("ij,j->i", -R, scene_center))
    return scale, transfm

if __name__ == "__main__":
    d = CasualDataset("v4", "/home/ubuntu/deform-colon/shape-of-motion/data/c3vd-v2/v4")
