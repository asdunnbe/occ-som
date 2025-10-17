from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from flow3d.configs import SurfaceModuleConfig


def _quat_xyzw_to_rotmat(quats: Tensor) -> Tensor:
    """
    Convert XYZW quaternions to rotation matrices.

    Args:
        quats: (..., 4) tensor with XYZW ordering.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    assert quats.shape[-1] == 4, "Quaternions must have shape (..., 4)"
    q = F.normalize(quats, dim=-1)
    x, y, z, w = torch.unbind(q, dim=-1)
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    rot = torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - zw),
            2 * (xz + yw),
            2 * (xy + zw),
            ww - xx + yy - zz,
            2 * (yz - xw),
            2 * (xz - yw),
            2 * (yz + xw),
            ww - xx - yy + zz,
        ],
        dim=-1,
    )
    return rot.reshape(*q.shape[:-1], 3, 3)


def _safe_exponential_weights(
    distances: Tensor, bandwidths: Tensor, eps: float
) -> Tensor:
    denom = 2.0 * torch.clamp(bandwidths**2, min=eps)
    return torch.exp(-torch.clamp(distances**2, min=0.0) / denom)


class SurfaceModule:
    """
    Mesh-free surface representation built from gaussian parameters using
    Moving-Least-Squares point-set surfaces and an intrinsic surface graph.
    """

    def __init__(self, cfg: SurfaceModuleConfig):
        self.cfg = cfg
        self._device: Optional[torch.device] = None
        self._surface_points: Optional[Tensor] = None
        self._surface_normals: Optional[Tensor] = None
        self._graph_indices: Optional[Tensor] = None
        self._graph_distances: Optional[Tensor] = None
        self._mls_neighbors: Optional[Tensor] = None
        self._mls_distances: Optional[Tensor] = None

    @property
    def surface_points(self) -> Optional[Tensor]:
        return self._surface_points

    @property
    def surface_normals(self) -> Optional[Tensor]:
        return self._surface_normals

    @property
    def graph_indices(self) -> Optional[Tensor]:
        return self._graph_indices

    @property
    def graph_distances(self) -> Optional[Tensor]:
        return self._graph_distances

    def update_from_gaussians(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
    ) -> None:
        """
        Update the MLS surface and intrinsic graph from gaussian parameters.
        """
        device = means.device
        num_points = means.shape[0]
        if num_points == 0:
            self._reset(device)
            return

        means_cpu = means.detach().float().cpu()
        quats_cpu = quats.detach().float().cpu()
        scales_cpu = scales.detach().float().cpu()

        normals_cpu = _quat_xyzw_to_rotmat(quats_cpu)[..., 2]
        bandwidths = torch.clamp(
            scales_cpu.mean(dim=-1) * self.cfg.bandwidth_scale,
            min=self.cfg.min_bandwidth,
        )

        knn_k = min(max(self.cfg.mls_neighbors, 2), num_points)
        if knn_k <= 1:
            self._reset(device)
            return

        indices, distances = self._build_knn(means_cpu, knn_k)
        (
            projected,
            proj_normals,
            weight_sums,
            neighbor_counts,
            max_weight_fracs,
        ) = self._project_points(
            means_cpu,
            normals_cpu,
            bandwidths,
            indices,
            distances,
        )
        valid_mask = torch.ones(projected.shape[0], dtype=torch.bool)
        if self.cfg.min_neighbor_weight > 0.0:
            valid_mask &= weight_sums >= self.cfg.min_neighbor_weight
        if self.cfg.min_neighbor_count > 0:
            valid_mask &= neighbor_counts >= self.cfg.min_neighbor_count
        if self.cfg.max_weight_fraction < 1.0:
            valid_mask &= max_weight_fracs <= self.cfg.max_weight_fraction

        if valid_mask.any():
            projected = projected[valid_mask]
            proj_normals = proj_normals[valid_mask]
            bandwidths = bandwidths[valid_mask]
        else:
            self._reset(device)
            return

        if projected.shape[0] == 0:
            self._reset(device)
            return

        decimation_radius = self.cfg.decimation_radius
        if decimation_radius < 0.0:
            decimation_radius = 0.0
        elif decimation_radius == 0.0:
            if bandwidths.numel() > 0:
                decimation_radius = float(torch.median(bandwidths).item())
            else:
                decimation_radius = 0.0

        if decimation_radius > 0.0:
            projected_cpu = projected
            proj_normals_cpu = proj_normals
            projected_cpu, proj_normals_cpu = self._voxel_downsample(
                projected_cpu, proj_normals_cpu, decimation_radius
            )
            projected = projected_cpu
            proj_normals = proj_normals_cpu

        num_surface_pts = projected.shape[0]
        if num_surface_pts == 0:
            self._reset(device)
            return

        self._surface_points = projected.to(device)
        self._surface_normals = proj_normals.to(device)
        self._device = device

        if self.cfg.graph_k > 0 and num_surface_pts > 1:
            graph_k = min(self.cfg.graph_k + 1, num_surface_pts)
            graph_indices, graph_distances = self._build_knn(projected, graph_k)
            if graph_indices.numel() > 0:
                self._graph_indices = graph_indices[:, 1:].to(torch.long)
                self._graph_distances = graph_distances[:, 1:]
            else:
                self._graph_indices = torch.empty(
                    (num_surface_pts, 0), dtype=torch.long, device=projected.device
                )
                self._graph_distances = torch.empty(
                    (num_surface_pts, 0), dtype=torch.float32, device=projected.device
                )
        else:
            self._graph_indices = torch.empty(
                (num_surface_pts, 0), dtype=torch.long, device=projected.device
            )
            self._graph_distances = torch.empty(
                (num_surface_pts, 0), dtype=torch.float32, device=projected.device
            )

        self._mls_neighbors = None
        self._mls_distances = None

    def _reset(self, device: torch.device) -> None:
        self._surface_points = torch.empty((0, 3), device=device)
        self._surface_normals = torch.empty((0, 3), device=device)
        self._graph_indices = torch.empty((0, 0), dtype=torch.long, device=device)
        self._graph_distances = torch.empty((0, 0), device=device)
        self._mls_neighbors = None
        self._mls_distances = None
        self._device = device

    @staticmethod
    def _build_knn(points: Tensor, k: int) -> tuple[Tensor, Tensor]:
        from sklearn.neighbors import NearestNeighbors

        np_points = points.numpy()
        knn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        knn.fit(np_points)
        distances_np, indices_np = knn.kneighbors(np_points)
        indices = torch.from_numpy(indices_np.astype(np.int64))
        distances = torch.from_numpy(distances_np.astype(np.float32))
        return indices, distances

    def _project_points(
        self,
        support_points: Tensor,
        support_normals: Tensor,
        bandwidths: Tensor,
        knn_indices: Tensor,
        initial_distances: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        eps = self.cfg.weight_epsilon

        neighbor_points = support_points[knn_indices]  # (N, K, 3)
        neighbor_normals = support_normals[knn_indices]  # (N, K, 3)
        neighbor_bandwidths = bandwidths[knn_indices]  # (N, K)

        projected = support_points.clone()
        normals = support_normals.clone()

        distances = initial_distances.clone()
        last_raw_weights = torch.zeros_like(distances)
        identity = torch.eye(3, dtype=torch.float32).unsqueeze(0)

        for _ in range(self.cfg.projection_iters):
            raw_weights = _safe_exponential_weights(distances, neighbor_bandwidths, eps)
            raw_weights = torch.nan_to_num(raw_weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights = raw_weights
            if self.cfg.weight_floor > 0.0:
                weights = weights + self.cfg.weight_floor
            weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
            weights = weights / weights_sum
            weighted_points = (weights[..., None] * neighbor_points).sum(dim=1)
            centroid = weighted_points

            centered = neighbor_points - centroid[:, None, :]
            weighted_centered = centered * weights[..., None]
            cov = torch.matmul(
                centered.transpose(1, 2),
                weighted_centered,
            )
            cov = cov + identity * self.cfg.covariance_reg

            evals, evecs = torch.linalg.eigh(cov)
            local_normals = evecs[:, :, 0]

            ref = (neighbor_normals * weights[..., None]).sum(dim=1)
            alignment = torch.sign((local_normals * ref).sum(dim=-1, keepdim=True))
            alignment[alignment == 0] = 1.0
            local_normals = local_normals * alignment

            offsets = ((projected - centroid) * local_normals).sum(
                dim=-1, keepdim=True
            )
            projected = projected - offsets * local_normals
            normals = local_normals

            distances = torch.linalg.norm(
                neighbor_points - projected[:, None, :], dim=-1
            )
            last_raw_weights = raw_weights

        weight_sums = last_raw_weights.sum(dim=-1)
        neighbor_counts = (last_raw_weights > eps).sum(dim=-1)
        max_weight_fracs = last_raw_weights.max(dim=-1).values / weight_sums.clamp_min(1e-8)

        return projected, normals, weight_sums, neighbor_counts, max_weight_fracs

    @staticmethod
    def _voxel_downsample(
        points: Tensor, normals: Tensor, radius: float
    ) -> tuple[Tensor, Tensor]:
        if radius <= 0.0 or points.shape[0] == 0:
            return points, normals

        coords = torch.floor(points / radius).to(torch.int64)
        unique_coords, inverse = torch.unique(coords, return_inverse=True, dim=0)
        num_voxels = unique_coords.shape[0]
        if num_voxels == points.shape[0]:
            return points, normals

        counts = torch.bincount(inverse, minlength=num_voxels).to(points.dtype)
        aggregated_points = torch.zeros((num_voxels, 3), dtype=points.dtype)
        aggregated_points.index_add_(0, inverse, points)
        aggregated_points = aggregated_points / counts.unsqueeze(-1)

        aggregated_normals = torch.zeros_like(aggregated_points)
        aggregated_normals.index_add_(0, inverse, normals)
        aggregated_normals = F.normalize(aggregated_normals, dim=-1, eps=1e-6)

        return aggregated_points, aggregated_normals

    def get_surface_points(self, device: Optional[torch.device] = None) -> Tensor:
        if self._surface_points is None:
            return torch.empty((0, 3), device=device or torch.device("cpu"))
        if device is None or self._surface_points.device == device:
            return self._surface_points
        return self._surface_points.to(device)

    def get_surface_normals(self, device: Optional[torch.device] = None) -> Tensor:
        if self._surface_normals is None:
            return torch.empty((0, 3), device=device or torch.device("cpu"))
        if device is None or self._surface_normals.device == device:
            return self._surface_normals
        return self._surface_normals.to(device)

    def get_graph(
        self, device: Optional[torch.device] = None
    ) -> tuple[Tensor, Tensor]:
        if self._graph_indices is None or self._graph_distances is None:
            empty_idx = torch.empty(
                (0, 0), dtype=torch.long, device=device or torch.device("cpu")
            )
            empty_dist = torch.empty((0, 0), device=device or torch.device("cpu"))
            return empty_idx, empty_dist

        indices = (
            self._graph_indices
            if device is None or self._graph_indices.device == device
            else self._graph_indices.to(device)
        )
        distances = (
            self._graph_distances
            if device is None or self._graph_distances.device == device
            else self._graph_distances.to(device)
        )
        return indices, distances

    def get_graph_segments(self, device: Optional[torch.device] = None) -> Tensor:
        indices, _ = self.get_graph(device=device)
        if indices.numel() == 0:
            target_device = device or indices.device
            return torch.empty((0, 2, 3), device=target_device)

        points = self.get_surface_points(device=indices.device)
        if points.numel() == 0:
            return torch.empty((0, 2, 3), device=indices.device)

        src = torch.arange(indices.shape[0], device=indices.device).unsqueeze(-1)
        src = src.expand_as(indices).reshape(-1)
        dst = indices.reshape(-1)
        mask = src < dst
        src = src[mask]
        dst = dst[mask]
        if src.numel() == 0:
            return torch.empty((0, 2, 3), device=points.device)
        segments = torch.stack([points[src], points[dst]], dim=1)
        if device is not None and segments.device != device:
            segments = segments.to(device)
        return segments
