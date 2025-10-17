import functools
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState
from pytorch_msssim import SSIM
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from flow3d.configs import (
    LossesConfig,
    OptimizerConfig,
    SceneLRConfig,
    SurfaceModuleConfig,
)
from flow3d.loss_utils import (
    compute_gradient_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    masked_l1_loss,
)
from flow3d.metrics import PCK, mLPIPS, mPSNR, mSSIM
from flow3d.scene_model import SceneModel
from flow3d.vis.utils import get_server
from flow3d.vis.viewer import DynamicViewer
from flow3d.normal_utils import depth_to_normal

class Trainer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        lr_cfg: SceneLRConfig,
        losses_cfg: LossesConfig,
        optim_cfg: OptimizerConfig,
        # Logging.
        work_dir: str,
        port: int | None = None,
        log_every: int = 10,
        checkpoint_every: int = 200,
        validate_every: int = 500,
        validate_video_every: int = 1000,
        validate_viewer_assets_every: int = 100,
        gaussian_knn_k: int | None = None,
        graph_save_every: int | None = None,
        viewer_graph_max_edges: int = 0,
        surface_cfg: SurfaceModuleConfig | None = None,
    ):
        self.device = device
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.validate_every = validate_every
        self.validate_video_every = validate_video_every
        self.validate_viewer_assets_every = validate_viewer_assets_every

        self.model = model
        self.num_frames = model.num_frames
        self.gaussian_knn_k = (
            gaussian_knn_k if gaussian_knn_k is not None and gaussian_knn_k > 0 else None
        )
        self._graph_initialized = False
        self.model.disable_knn_graph()

        if graph_save_every is None:
            self.graph_save_every = checkpoint_every
        else:
            self.graph_save_every = int(graph_save_every)
        if self.graph_save_every < 0:
            self.graph_save_every = 0
        self.graph_save_dir = Path(work_dir) / "graphs"
        self._last_saved_graph_step = -1

        self.viewer_graph_max_edges = max(0, int(viewer_graph_max_edges))
        self._viewer_graph_handles: list[Any] = []
        self._graph_rng = np.random.default_rng()

        self.surface_cfg = surface_cfg or SurfaceModuleConfig()
        self.surface_enabled = bool(self.surface_cfg.enabled)
        self.surface_save_dir = Path(work_dir) / "surface"
        self._surface_last_graph_step = -1
        self._surface_initialized = False

        self.lr_cfg = lr_cfg
        self.losses_cfg = losses_cfg
        self.optim_cfg = optim_cfg

        self.reset_opacity_every = (
            self.optim_cfg.reset_opacity_every_n_controls * self.optim_cfg.control_every
        )
        self.optimizers, self.scheduler = self.configure_optimizers()

        # running stats for adaptive density control
        self.running_stats = {
            "xys_grad_norm_acc": torch.zeros(self.model.num_gaussians, device=device),
            "vis_count": torch.zeros(
                self.model.num_gaussians, device=device, dtype=torch.int64
            ),
            "max_radii": torch.zeros(self.model.num_gaussians, device=device),
        }

        self.work_dir = work_dir
        self.writer = SummaryWriter(log_dir=work_dir)
        self.global_step = 0
        self.epoch = 0

        self._maybe_initialize_graph()
        self._maybe_initialize_surface()

        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server, self.render_fn, model.num_frames, work_dir, mode="training"
            )

        # metrics
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.psnr_metric = mPSNR()
        self.ssim_metric = mSSIM()
        self.lpips_metric = mLPIPS()
        self.pck_metric = PCK()
        self.bg_psnr_metric = mPSNR()
        self.fg_psnr_metric = mPSNR()
        self.bg_ssim_metric = mSSIM()
        self.fg_ssim_metric = mSSIM()
        self.bg_lpips_metric = mLPIPS()
        self.fg_lpips_metric = mLPIPS()

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def save_checkpoint(self, path: str):
        model_dict = self.model.state_dict()
        optimizer_dict = {k: v.state_dict() for k, v in self.optimizers.items()}
        scheduler_dict = {k: v.state_dict() for k, v in self.scheduler.items()}
        ckpt = {
            "model": model_dict,
            "optimizers": optimizer_dict,
            "schedulers": scheduler_dict,
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        torch.save(ckpt, path)
        guru.info(f"Saved checkpoint at {self.global_step=} to {path}")

    @staticmethod
    def init_from_checkpoint(
        path: str,
        device: torch.device,
        use_2dgs,
        *args,
        surface_cfg: SurfaceModuleConfig | None = None,
        **kwargs,
    ) -> tuple["Trainer", int]:
        guru.info(f"Loading checkpoint from {path}")
        try:
            ckpt = torch.load(path, weights_only=False)
        except TypeError:
            ckpt = torch.load(path)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)
        print(use_2dgs)
        model.use_2dgs = use_2dgs
        trainer = Trainer(model, device, *args, surface_cfg=surface_cfg, **kwargs)
        if "optimizers" in ckpt:
            trainer.load_checkpoint_optimizers(ckpt["optimizers"])
        if "schedulers" in ckpt:
            trainer.load_checkpoint_schedulers(ckpt["schedulers"])
        trainer.global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        trainer.set_epoch(start_epoch)
        return trainer, start_epoch

    def load_checkpoint_optimizers(self, opt_ckpt):
        for k, v in self.optimizers.items():
            v.load_state_dict(opt_ckpt[k])

    def load_checkpoint_schedulers(self, sched_ckpt):
        for k, v in self.scheduler.items():
            v.load_state_dict(sched_ckpt[k])

    @torch.inference_mode()
    def render_fn(
        self,
        camera_state: CameraState,
        render_state,  # RenderTabState in newer nerfview, Tuple[int, int] in older releases.
    ):
        if hasattr(render_state, "viewer_width") and hasattr(
            render_state, "viewer_height"
        ):
            W = int(render_state.viewer_width)
            H = int(render_state.viewer_height)
            img_wh = (W, H)
        else:
            W, H = render_state
            img_wh = render_state

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = 0
        if self.viewer is not None and hasattr(self.viewer, "_playback_guis"):
            canonical_checkbox = getattr(self.viewer, "_canonical_checkbox", None)
            if canonical_checkbox is not None and canonical_checkbox.value:
                t = None
            else:
                t = int(self.viewer._playback_guis[0].value)
        self.model.training = False
        img = self.model.render(t, w2c[None], K[None], img_wh)["img"][0]
        return (img.cpu().numpy() * 255.0).astype(np.uint8)

    def train_step(self, batch):
        if self.viewer is not None:
            while True:
                viewer_state = getattr(self.viewer, "state", None)
                if hasattr(viewer_state, "status"):
                    is_paused = viewer_state.status == "paused"
                else:
                    is_paused = viewer_state == "paused"
                if not is_paused:
                    break
                time.sleep(0.1)
            self.viewer.lock.acquire()

        loss, stats, num_rays_per_step, num_rays_per_sec = self.compute_losses(batch)
        if loss.isnan():
            guru.info(f"Loss is NaN at step {self.global_step}!!")
            import ipdb

            ipdb.set_trace()
        loss.backward()

        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if self._graph_initialized:
            self.model.mark_knn_graph_dirty()
        for sched in self.scheduler.values():
            sched.step()

        self.log_dict(stats)
        self.global_step += 1
        self._maybe_initialize_graph()
        self._maybe_initialize_surface()
        self.run_control_steps()

        should_save_graph = (
            self._graph_initialized
            and self.graph_save_every > 0
            and self.global_step > 0
            and self.global_step % self.graph_save_every == 0
            and self.global_step != self._last_saved_graph_step
        )
        should_update_viewer_graph = (
            self.viewer is not None
            and self._graph_initialized
            and self.viewer_graph_max_edges > 0
            and self.validate_viewer_assets_every > 0
            and self.global_step > 0
            and self.global_step % self.validate_viewer_assets_every == 0
        )

        if should_save_graph or should_update_viewer_graph:
            self.model.rebuild_knn_graph()
        if should_save_graph:
            self.save_graph_snapshot()

        if self.surface_enabled and self._surface_initialized:
            surface_update_due = (
                self.surface_cfg.graph_update_every > 0
                and self.global_step > 0
                and self.global_step % self.surface_cfg.graph_update_every == 0
            ) or should_update_viewer_graph
            surface_snapshot_due = (
                self.surface_cfg.snapshot_every > 0
                and self.global_step > 0
                and self.global_step % self.surface_cfg.snapshot_every == 0
            )
            if (surface_update_due or surface_snapshot_due) and (
                self._surface_last_graph_step != self.global_step
            ):
                self.model.update_surface_module(force=True)
                self._surface_last_graph_step = self.global_step
            if surface_snapshot_due:
                self.save_surface_snapshot()

        if self.viewer is not None:
            self.viewer.lock.release()
            viewer_state = getattr(self.viewer, "state", None)
            if hasattr(viewer_state, "num_train_rays_per_sec"):
                viewer_state.num_train_rays_per_sec = num_rays_per_sec
            if self.viewer.mode == "training":
                self.viewer.update(self.global_step, num_rays_per_step)
            if should_update_viewer_graph:
                self._update_viewer_graph()

        if self.global_step % self.checkpoint_every == 0:
            self.save_checkpoint(f"{self.work_dir}/checkpoints/last.ckpt")

        return loss.item()

    def compute_losses(self, batch):
        self.model.training = True

        B = batch["imgs"].shape[0]
        W, H = img_wh = batch["imgs"].shape[2:0:-1]
        N = batch["target_ts"][0].shape[0]

        # (B,).
        ts = batch["ts"]
        # (B, 4, 4).
        w2cs = batch["w2cs"]
        # (B, 3, 3).
        Ks = batch["Ks"]
        # (B, H, W, 3).
        imgs = batch["imgs"]
        # (B, H, W).
        valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0]))
        # (B, H, W).
        masks = batch["masks"]
        masks *= valid_masks
        # (B, H, W).
        depths = batch["depths"]
        # (B, H, W, 3)
        try:
            normals = batch["normals"]
        except:
            pass
        # [(P, 2), ...].
        query_tracks_2d = batch["query_tracks_2d"]
        # [(N,), ...].
        target_ts = batch["target_ts"]
        # [(N, 4, 4), ...].
        target_w2cs = batch["target_w2cs"]
        # [(N, 3, 3), ...].
        target_Ks = batch["target_Ks"]
        # [(N, P, 2), ...].
        target_tracks_2d = batch["target_tracks_2d"]
        # [(N, P), ...].
        target_visibles = batch["target_visibles"]
        # [(N, P), ...].
        target_invisibles = batch["target_invisibles"]
        # [(N, P), ...].
        target_confidences = batch["target_confidences"]
        # [(N, P), ...].
        target_track_depths = batch["target_track_depths"]

        _tic = time.time()
        # (B, G, 3).
        means, quats = self.model.compute_poses_all(ts)  # (G, B, 3), (G, B, 4)
        device = means.device
        means = means.transpose(0, 1)
        quats = quats.transpose(0, 1)
        # [(N, G, 3), ...].
        target_ts_vec = torch.cat(target_ts)
        # (B * N, G, 3).
        target_means, _ = self.model.compute_poses_all(target_ts_vec)
        target_means = target_means.transpose(0, 1)
        target_mean_list = target_means.split(N)
        num_frames = self.model.num_frames

        loss = 0.0

        bg_colors = []
        rendered_all = []
        self._batched_xys = []
        self._batched_radii = []
        self._batched_img_wh = []
        for i in range(B):
            bg_color = torch.ones(1, 3, device=device)
            rendered = self.model.render(
                ts[i].item(),
                w2cs[None, i],
                Ks[None, i],
                img_wh,
                target_ts=target_ts[i],
                target_w2cs=target_w2cs[i],
                bg_color=bg_color,
                means=means[i],
                quats=quats[i],
                target_means=target_mean_list[i].transpose(0, 1),
                return_depth=True,
                return_mask=self.model.has_bg,
            )
            rendered_all.append(rendered)
            bg_colors.append(bg_color)
            if (
                self.model._current_xys is not None
                and self.model._current_radii is not None
                and self.model._current_img_wh is not None
            ):
                self._batched_xys.append(self.model._current_xys)
                self._batched_radii.append(self.model._current_radii)
                self._batched_img_wh.append(self.model._current_img_wh)

        # Necessary to make viewer work.
        num_rays_per_step = H * W * B
        num_rays_per_sec = num_rays_per_step / (time.time() - _tic)

        # (B, H, W, N, *).
        rendered_all = {
            key: (
                torch.cat([out_dict[key] for out_dict in rendered_all], dim=0)
                if rendered_all[0][key] is not None
                else None
            )
            for key in rendered_all[0]
        }
        bg_colors = torch.cat(bg_colors, dim=0)

        # Compute losses.
        # (B * N).
        frame_intervals = (ts.repeat_interleave(N) - target_ts_vec).abs()
        if not self.model.has_bg:
            imgs = (
                imgs * masks[..., None]
                + (1.0 - masks[..., None]) * bg_colors[:, None, None]
            )
        else:
            imgs = (
                imgs * valid_masks[..., None]
                + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
            )
        # (P_all, 2).
        tracks_2d = torch.cat([x.reshape(-1, 2) for x in target_tracks_2d], dim=0)
        # (P_all,)
        visibles = torch.cat([x.reshape(-1) for x in target_visibles], dim=0)
        # (P_all,)
        confidences = torch.cat([x.reshape(-1) for x in target_confidences], dim=0)

        if rendered_all["rend_normal"] != None and rendered_all["surf_normal"] != None:
            # 2DGS normal consistency
            rendered_normals = cast(torch.Tensor, rendered_all["rend_normal"])
            surf_normals = cast(torch.Tensor, rendered_all["surf_normal"])
            surf_normals = surf_normals.reshape(rendered_normals.shape)
            cos_sim = torch.sum(rendered_normals * surf_normals, dim=-1)
            normal_loss = (1 - cos_sim).mean()
            loss += normal_loss * 0.05                  # NOTE: small normal loss weight


        # RGB loss.
        rendered_imgs = cast(torch.Tensor, rendered_all["img"])
        if self.model.has_bg:
            rendered_imgs = (
                rendered_imgs * valid_masks[..., None]
                + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
            )
        rgb_loss = 0.8 * F.l1_loss(rendered_imgs, imgs) + 0.2 * (
            1 - self.ssim(rendered_imgs.permute(0, 3, 1, 2), imgs.permute(0, 3, 1, 2))
        )
        loss += rgb_loss * self.losses_cfg.w_rgb

        # Mask loss.
        if not self.model.has_bg:
            mask_loss = F.mse_loss(rendered_all["acc"], masks[..., None])  # type: ignore
        else:
            mask_loss = F.mse_loss(
                rendered_all["acc"], torch.ones_like(rendered_all["acc"])  # type: ignore
            ) + masked_l1_loss(
                rendered_all["mask"],
                masks[..., None],
                quantile=0.98,  # type: ignore
            )
        loss += mask_loss * self.losses_cfg.w_mask

        # (B * N, H * W, 3).
        pred_tracks_3d = (
            rendered_all["tracks_3d"].permute(0, 3, 1, 2, 4).reshape(-1, H * W, 3)  # type: ignore
        )
        pred_tracks_2d = torch.einsum(
            "bij,bpj->bpi", torch.cat(target_Ks), pred_tracks_3d
        )
        # (B * N, H * W, 1).
        mapped_depth = torch.clamp(pred_tracks_2d[..., 2:], min=1e-6)
        # (B * N, H * W, 2).
        pred_tracks_2d = pred_tracks_2d[..., :2] / mapped_depth

        # (B * N).
        w_interval = torch.exp(-2 * frame_intervals / num_frames)
        # w_track_loss = min(1, (self.max_steps - self.global_step) / 6000)
        track_weights = confidences[..., None] * w_interval

        # (B, H, W).
        masks_flatten = torch.zeros_like(masks)
        for i in range(B):
            # This takes advantage of the fact that the query 2D tracks are
            # always on the grid.
            query_pixels = query_tracks_2d[i].to(torch.int64)
            masks_flatten[i, query_pixels[:, 1], query_pixels[:, 0]] = 1.0
        # (B * N, H * W).
        masks_flatten = (
            masks_flatten.reshape(-1, H * W).tile(1, N).reshape(-1, H * W) > 0.5
        )

        track_2d_loss = masked_l1_loss(
            pred_tracks_2d[masks_flatten][visibles],
            tracks_2d[visibles],
            mask=track_weights[visibles],
            quantile=0.98,
        ) / max(H, W)
        loss += track_2d_loss * self.losses_cfg.w_track

        depth_masks = (
            masks[..., None] if not self.model.has_bg else valid_masks[..., None]
        )

        pred_depth = cast(torch.Tensor, rendered_all["depth"])
        pred_disp = 1.0 / (pred_depth + 1e-5)
        tgt_disp = 1.0 / (depths[..., None] + 1e-5)
        depth_loss = masked_l1_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks,
            quantile=0.98,
        )
        loss += depth_loss * self.losses_cfg.w_depth_reg

        # mapped depth loss (using cached depth with EMA)
        #  mapped_depth_loss = 0.0
        mapped_depth_gt = torch.cat([x.reshape(-1) for x in target_track_depths], dim=0)
        mapped_depth_loss = masked_l1_loss(
            1 / (mapped_depth[masks_flatten][visibles] + 1e-5),
            1 / (mapped_depth_gt[visibles, None] + 1e-5),
            track_weights[visibles],
        )

        loss += mapped_depth_loss * self.losses_cfg.w_depth_const

        #  depth_gradient_loss = 0.0
        depth_gradient_loss = compute_gradient_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks > 0.5,
            quantile=0.95,
        )
        loss += depth_gradient_loss * self.losses_cfg.w_depth_grad

        # bases should be smooth.
        small_accel_loss = compute_se3_smoothness_loss(
            self.model.motion_bases.params["rots"],
            self.model.motion_bases.params["transls"],
        )
        loss += small_accel_loss * self.losses_cfg.w_smooth_bases


# ======= Eucidian space constraints =======
        # tracks should be smooth
        ts = torch.clamp(ts, min=1, max=num_frames - 2)
        ts_neighbors = torch.cat((ts - 1, ts, ts + 1))
        transfms_nbs = self.model.compute_transforms(ts_neighbors)  # (G, 3n, 3, 4)
        means_fg_nbs = torch.einsum(
            "pnij,pj->pni",
            transfms_nbs,
            F.pad(self.model.fg.params["means"], (0, 1), value=1.0),
        )
        means_fg_nbs = means_fg_nbs.reshape(
            means_fg_nbs.shape[0], 3, -1, 3
        )  # [G, 3, n, 3]
        if self.losses_cfg.w_smooth_tracks > 0:
            small_accel_loss_tracks = 0.5 * (
                (2 * means_fg_nbs[:, 1:-1] - means_fg_nbs[:, :-2] - means_fg_nbs[:, 2:])
                .norm(dim=-1)
                .mean()
            )
            loss += small_accel_loss_tracks * self.losses_cfg.w_smooth_tracks

# ========================================

        # Constrain the std of scales.
        # TODO: do we want to penalize before or after exp?
        loss += (
            self.losses_cfg.w_scale_var
            * torch.var(torch.exp(self.model.fg.params["scales"]), dim=-1).mean()
        )
        if self.model.bg is not None:
            loss += (
                self.losses_cfg.w_scale_var
                * torch.var(torch.exp(self.model.bg.params["scales"]), dim=-1).mean()
            )
        
        if self.model.fg.params["means"].isnan().sum() > 0:
            import ipdb
            ipdb.set_trace()
        # # sparsity loss
        # loss += 0.01 * self.opacity_activation(self.opacities).abs().mean()

        # Acceleration along ray direction should be small.
        z_accel_loss = compute_z_acc_loss(means_fg_nbs, w2cs)


        loss += self.losses_cfg.w_z_accel * z_accel_loss

        # Prepare stats for logging.
        stats = {
            "train/loss": loss.item(),
            "train/rgb_loss": rgb_loss.item(),
            "train/mask_loss": mask_loss.item(),
            "train/depth_loss": depth_loss.item(),
            "train/depth_gradient_loss": depth_gradient_loss.item(),
            "train/mapped_depth_loss": mapped_depth_loss.item(),
            "train/track_2d_loss": track_2d_loss.item(),
            "train/small_accel_loss": small_accel_loss.item(),
            "train/z_acc_loss": z_accel_loss.item(),
            "train/num_gaussians": self.model.num_gaussians,
            "train/num_fg_gaussians": self.model.num_fg_gaussians,
            "train/num_bg_gaussians": self.model.num_bg_gaussians,
        }

        # Compute metrics.
        with torch.no_grad():
            psnr = self.psnr_metric(
                rendered_imgs, imgs, masks if not self.model.has_bg else valid_masks
            )
            self.psnr_metric.reset()
            stats["train/psnr"] = psnr
            if self.model.has_bg:
                bg_psnr = self.bg_psnr_metric(rendered_imgs, imgs, 1.0 - masks)
                fg_psnr = self.fg_psnr_metric(rendered_imgs, imgs, masks)
                self.bg_psnr_metric.reset()
                self.fg_psnr_metric.reset()
                stats["train/bg_psnr"] = bg_psnr
                stats["train/fg_psnr"] = fg_psnr

        stats.update(
            **{
                "train/num_rays_per_sec": num_rays_per_sec,
                "train/num_rays_per_step": float(num_rays_per_step),
            }
        )

        return loss, stats, num_rays_per_step, num_rays_per_sec

    def log_dict(self, stats: dict):
        for k, v in stats.items():
            self.writer.add_scalar(k, v, self.global_step)

    def _maybe_initialize_graph(self) -> None:
        if self._graph_initialized or self.gaussian_knn_k is None:
            return
        warmup_steps = getattr(self.optim_cfg, "warmup_steps", 0)
        if self.global_step < warmup_steps:
            return
        self.model.enable_knn_graph(self.gaussian_knn_k)
        self._graph_initialized = True
        guru.info(
            f"Initialized Gaussian K-NN graph with k={self.gaussian_knn_k} at step {self.global_step}."
        )

    def _maybe_initialize_surface(self) -> None:
        if not self.surface_enabled or self._surface_initialized:
            return
        warmup_steps = getattr(self.optim_cfg, "warmup_steps", 0)
        if self.global_step < warmup_steps:
            return
        self.model.init_surface_module(self.surface_cfg)
        self._surface_initialized = True
        guru.info(
            f"Initialized surface module at step {self.global_step} "
            f"(graph_k={self.surface_cfg.graph_k})."
        )

    def _refresh_knn_graph(self) -> None:
        if not self._graph_initialized:
            return
        self.model.rebuild_knn_graph()

    def save_graph_snapshot(self) -> None:
        if not self._graph_initialized:
            return
        indices, distances = self.model.get_knn_graph()
        if indices is None or distances is None:
            return
        means = self.model.get_canonical_means_all().detach().cpu()
        snapshot = {
            "step": int(self.global_step),
            "k": int(self.gaussian_knn_k),
            "indices": indices.cpu(),
            "distances": distances.cpu(),
            "means": means,
        }
        self.graph_save_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.graph_save_dir / f"graph_step_{self.global_step:08d}.pt"
        torch.save(snapshot, out_path)
        self._last_saved_graph_step = self.global_step
        guru.info(f"Saved gaussian graph snapshot to {out_path}")

    def save_surface_snapshot(self) -> None:
        if (
            not self.surface_enabled
            or not self._surface_initialized
            or self.model.surface_module is None
        ):
            return
        points = self.model.get_surface_points().detach().cpu()
        normals = self.model.get_surface_normals().detach().cpu()
        indices, distances = self.model.get_surface_graph()
        indices = indices.detach().cpu()
        distances = distances.detach().cpu()
        snapshot = {
            "step": int(self.global_step),
            "graph_k": int(self.surface_cfg.graph_k),
            "points": points,
            "normals": normals,
            "indices": indices,
            "distances": distances,
        }
        self.surface_save_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.surface_save_dir / f"surface_step_{self.global_step:08d}.pt"
        torch.save(snapshot, out_path)
        guru.info(f"Saved surface snapshot to {out_path}")

    def _clear_viewer_graph(self) -> None:
        if not self._viewer_graph_handles:
            return
        for handle in self._viewer_graph_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._viewer_graph_handles = []

    def _update_viewer_graph(self) -> None:
        if (
            self.viewer is None
            or self.viewer_graph_max_edges <= 0
            or not self._graph_initialized
        ):
            return
        server = getattr(self.viewer, "server", None)
        if server is None:
            return

        gaussian_segments = self.model.get_knn_segments()
        surface_segments = torch.empty((0, 2, 3), device=gaussian_segments.device)
        if (
            self.surface_enabled
            and self._surface_initialized
            and self.model.surface_module is not None
        ):
            surface_segments = self.model.get_surface_segments()

        gaussian_np = (
            gaussian_segments.detach().cpu().numpy()
            if gaussian_segments.numel() > 0
            else np.empty((0, 2, 3), dtype=np.float32)
        )
        surface_np = (
            surface_segments.detach().cpu().numpy()
            if surface_segments.numel() > 0
            else np.empty((0, 2, 3), dtype=np.float32)
        )

        if gaussian_np.shape[0] == 0 and surface_np.shape[0] == 0:
            self._clear_viewer_graph()
            return

        max_edges = max(0, self.viewer_graph_max_edges)
        if gaussian_np.shape[0] > 0 and surface_np.shape[0] > 0 and max_edges > 0:
            gaussian_limit = max(1, max_edges // 2)
            surface_limit = max(0, max_edges - gaussian_limit)
        elif gaussian_np.shape[0] > 0:
            gaussian_limit = max_edges
            surface_limit = 0
        else:
            gaussian_limit = 0
            surface_limit = max_edges

        if gaussian_limit > 0 and gaussian_np.shape[0] > gaussian_limit:
            choice = self._graph_rng.choice(
                gaussian_np.shape[0], size=gaussian_limit, replace=False
            )
            gaussian_np = gaussian_np[choice]
        if surface_limit > 0 and surface_np.shape[0] > surface_limit:
            choice = self._graph_rng.choice(
                surface_np.shape[0], size=surface_limit, replace=False
            )
            surface_np = surface_np[choice]
        if gaussian_limit == 0:
            gaussian_np = gaussian_np[:0]
        if surface_limit == 0:
            surface_np = surface_np[:0]

        gaussian_weights = (
            np.linalg.norm(gaussian_np[:, 0] - gaussian_np[:, 1], axis=-1)
            if gaussian_np.size > 0
            else np.empty(0, dtype=np.float32)
        )
        surface_weights = (
            np.linalg.norm(surface_np[:, 0] - surface_np[:, 1], axis=-1)
            if surface_np.size > 0
            else np.empty(0, dtype=np.float32)
        )
        if gaussian_weights.size > 0:
            gaussian_weights = (gaussian_weights - gaussian_weights.min()) / (
                gaussian_weights.ptp() + 1e-8
            )
        if surface_weights.size > 0:
            surface_weights = (surface_weights - surface_weights.min()) / (
                surface_weights.ptp() + 1e-8
            )

        self._clear_viewer_graph()
        handles: list[Any] = []

        node_positions = self.model.get_canonical_means_all().detach().cpu().numpy()
        if node_positions.shape[0] > 0:
            node_colors = np.tile(
                np.array([[0.2, 0.8, 1.0]], dtype=np.float32),
                (node_positions.shape[0], 1),
            )
            handles.append(
                server.scene.add_point_cloud(
                    "/graph/gaussian_nodes",
                    points=node_positions,
                    colors=node_colors,
                    point_size=0.01,
                )
            )

        if (
            self.surface_enabled
            and self._surface_initialized
            and self.model.surface_module is not None
        ):
            surface_pts_tensor = self.model.get_surface_points()
            if surface_pts_tensor.numel() > 0:
                surface_nodes = surface_pts_tensor.detach().cpu().numpy()
                surface_colors = np.tile(
                    np.array([[0.95, 0.55, 0.2]], dtype=np.float32),
                    (surface_nodes.shape[0], 1),
                )
                handles.append(
                    server.scene.add_point_cloud(
                        "/graph/surface_nodes",
                        points=surface_nodes,
                        colors=surface_colors,
                        point_size=0.008,
                    )
                )

        for idx, (segment, wn) in enumerate(zip(gaussian_np, gaussian_weights)):
            color = (float(wn), float(1.0 - wn), 0.2)
            handles.append(
                server.scene.add_spline_catmull_rom(
                    f"/graph/gaussian_edges/{idx}",
                    positions=segment,
                    color=color,
                    segments=1,
                    line_width=0.002,
                )
            )

        for idx, (segment, wn) in enumerate(zip(surface_np, surface_weights)):
            color = (0.95, float(0.2 + 0.6 * (1.0 - wn)), 0.25)
            handles.append(
                server.scene.add_spline_catmull_rom(
                    f"/graph/surface_edges/{idx}",
                    positions=segment,
                    color=color,
                    segments=1,
                    line_width=0.002,
                )
            )

        self._viewer_graph_handles = handles

    def run_control_steps(self):
        global_step = self.global_step
        # Adaptive gaussian control.
        cfg = self.optim_cfg
        num_frames = self.model.num_frames
        ready = self._prepare_control_step()
        if (
            ready
            and global_step > cfg.warmup_steps
            and global_step % cfg.control_every == 0
            and global_step < cfg.stop_control_steps
        ):
            if (
                global_step < cfg.stop_densify_steps
                and global_step % self.reset_opacity_every > num_frames
            ):
                self._densify_control_step(global_step)
            if global_step % self.reset_opacity_every > min(3 * num_frames, 1000):
                self._cull_control_step(global_step)
            if global_step % self.reset_opacity_every == 0:
                self._reset_opacity_control_step()

            # Reset stats after every control.
            for k in self.running_stats:
                self.running_stats[k].zero_()

    @torch.no_grad()
    def _prepare_control_step(self) -> bool:
        # Prepare for adaptive gaussian control based on the current stats.
        if not (
            self.model._current_radii is not None
            and self.model._current_xys is not None
        ):
            guru.warning("Model not training, skipping control step preparation")
            return False

        batch_size = len(self._batched_xys)
        # these quantities are for each rendered view and have shapes (C, G, *)
        # must be aggregated over all views
        for _current_xys, _current_radii, _current_img_wh in zip(
            self._batched_xys, self._batched_radii, self._batched_img_wh
        ):
            radii_tensor = _current_radii
            xys_tensor = _current_xys

            if radii_tensor.ndim == 2:
                radii_tensor = radii_tensor.unsqueeze(0)
                xys_tensor = xys_tensor.unsqueeze(0)

            if isinstance(_current_img_wh, torch.Tensor):
                if _current_img_wh.ndim == 0:
                    img_wh_tensor = _current_img_wh.reshape(1, 1)
                elif _current_img_wh.ndim == 1:
                    img_wh_tensor = _current_img_wh.unsqueeze(0)
                else:
                    img_wh_tensor = _current_img_wh
            else:
                img_wh_tensor = _current_img_wh

            num_views = radii_tensor.shape[0]
            for view_idx in range(num_views):
                radii_view = radii_tensor[view_idx]
                xys_grad_view = xys_tensor.grad[view_idx]

                if isinstance(img_wh_tensor, torch.Tensor):
                    wh_view = img_wh_tensor[view_idx] if img_wh_tensor.ndim > 1 else img_wh_tensor[0]
                    width = float(wh_view[0].item()) if wh_view.ndim > 0 else float(wh_view.item())
                    height = float(wh_view[1].item()) if wh_view.ndim > 0 else float(wh_view.item())
                elif isinstance(img_wh_tensor, (list, tuple)):
                    entry = img_wh_tensor[view_idx] if len(img_wh_tensor) > 2 and isinstance(img_wh_tensor[0], (list, tuple, torch.Tensor)) else img_wh_tensor
                    width, height = entry
                else:
                    width = height = float(_current_img_wh)

                visibility_axes = radii_view > 0
                visible_gaussians = visibility_axes.any(dim=-1)
                gidcs = torch.where(visible_gaussians)[0]
                if gidcs.numel() == 0:
                    continue

                xys_grad_view[..., 0] *= width / 2.0 * batch_size
                xys_grad_view[..., 1] *= height / 2.0 * batch_size

                grad_norm = xys_grad_view[visible_gaussians].norm(dim=-1)
                self.running_stats["xys_grad_norm_acc"].index_add_(0, gidcs, grad_norm)
                self.running_stats["vis_count"].index_add_(
                    0, gidcs, torch.ones_like(gidcs, dtype=torch.int64)
                )

                radii_vals = radii_view[visible_gaussians].amax(dim=-1)
                max_radii = torch.maximum(
                    self.running_stats["max_radii"].index_select(0, gidcs),
                    radii_vals / max(width, height),
                )
                self.running_stats["max_radii"].index_put((gidcs,), max_radii)
        return True

    @torch.no_grad()
    def _densify_control_step(self, global_step):
        assert (self.running_stats["vis_count"] > 0).any()

        cfg = self.optim_cfg
        xys_grad_avg = self.running_stats["xys_grad_norm_acc"] / self.running_stats[
            "vis_count"
        ].clamp_min(1)
        is_grad_too_high = xys_grad_avg > cfg.densify_xys_grad_threshold
        # Split gaussians.
        scales = self.model.get_scales_all()
        is_scale_too_big = scales.amax(dim=-1) > cfg.densify_scale_threshold
        if global_step < cfg.stop_control_by_screen_steps:
            is_radius_too_big = (
                self.running_stats["max_radii"] > cfg.densify_screen_threshold
            )
        else:
            is_radius_too_big = torch.zeros_like(is_grad_too_high, dtype=torch.bool)

        should_split = is_grad_too_high & (is_scale_too_big | is_radius_too_big)
        should_dup = is_grad_too_high & ~is_scale_too_big

        num_fg = self.model.num_fg_gaussians
        should_fg_split = should_split[:num_fg]
        num_fg_splits = int(should_fg_split.sum().item())
        should_fg_dup = should_dup[:num_fg]
        num_fg_dups = int(should_fg_dup.sum().item())

        should_bg_split = should_split[num_fg:]
        num_bg_splits = int(should_bg_split.sum().item())
        should_bg_dup = should_dup[num_fg:]
        num_bg_dups = int(should_bg_dup.sum().item())

        fg_param_map = self.model.fg.densify_params(should_fg_split, should_fg_dup)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            dup_in_optim(
                optimizer,
                [new_params],
                should_fg_split,
                num_fg_splits * 2 + num_fg_dups,
            )

        if self.model.bg is not None:
            bg_param_map = self.model.bg.densify_params(should_bg_split, should_bg_dup)
            for param_name, new_params in bg_param_map.items():
                full_param_name = f"bg.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                dup_in_optim(
                    optimizer,
                    [new_params],
                    should_bg_split,
                    num_bg_splits * 2 + num_bg_dups,
                )

        # update running stats
        for k, v in self.running_stats.items():
            v_fg, v_bg = v[:num_fg], v[num_fg:]
            new_v = torch.cat(
                [
                    v_fg[~should_fg_split],
                    v_fg[should_fg_dup],
                    v_fg[should_fg_split].repeat(2),
                    v_bg[~should_bg_split],
                    v_bg[should_bg_dup],
                    v_bg[should_bg_split].repeat(2),
                ],
                dim=0,
            )
            self.running_stats[k] = new_v
        self._refresh_knn_graph()
        guru.info(
            f"Split {should_split.sum().item()} gaussians, "
            f"Duplicated {should_dup.sum().item()} gaussians, "
            f"{self.model.num_gaussians} gaussians left"
        )

    @torch.no_grad()
    def _cull_control_step(self, global_step):
        # Cull gaussians.
        cfg = self.optim_cfg
        opacities = self.model.get_opacities_all()
        device = opacities.device
        is_opacity_too_small = opacities < cfg.cull_opacity_threshold
        is_radius_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        is_scale_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        cull_scale_threshold = (
            torch.ones(len(is_scale_too_big), device=device) * cfg.cull_scale_threshold
        )
        num_fg = self.model.num_fg_gaussians
        cull_scale_threshold[num_fg:] *= self.model.bg_scene_scale
        if global_step > self.reset_opacity_every:
            scales = self.model.get_scales_all()
            is_scale_too_big = scales.amax(dim=-1) > cull_scale_threshold
            if global_step < cfg.stop_control_by_screen_steps:
                is_radius_too_big = (
                    self.running_stats["max_radii"] > cfg.cull_screen_threshold
                )
        should_cull = is_opacity_too_small | is_radius_too_big | is_scale_too_big
        should_fg_cull = should_cull[:num_fg]
        should_bg_cull = should_cull[num_fg:]

        fg_param_map = self.model.fg.cull_params(should_fg_cull)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            remove_from_optim(optimizer, [new_params], should_fg_cull)

        if self.model.bg is not None:
            bg_param_map = self.model.bg.cull_params(should_bg_cull)
            for param_name, new_params in bg_param_map.items():
                full_param_name = f"bg.params.{param_name}"
                optimizer = self.optimizers[full_param_name]
                remove_from_optim(optimizer, [new_params], should_bg_cull)

        # update running stats
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[~should_cull]

        self._refresh_knn_graph()
        guru.info(
            f"Culled {should_cull.sum().item()} gaussians, "
            f"{self.model.num_gaussians} gaussians left"
        )

    @torch.no_grad()
    def _reset_opacity_control_step(self):
        # Reset gaussian opacities.
        new_val = torch.logit(torch.tensor(0.8 * self.optim_cfg.cull_opacity_threshold))
        for part in ["fg", "bg"]:
            module = getattr(self.model, part, None)
            if module is None:
                continue
            part_params = module.reset_opacities(new_val)
            # Modify optimizer states by new assignment.
            for param_name, new_params in part_params.items():
                full_param_name = f"{part}.params.{param_name}"
                optimizer = self.optimizers.get(full_param_name)
                if optimizer is None:
                    continue
                reset_in_optim(optimizer, [new_params])
        guru.info("Reset opacities")

    def configure_optimizers(self):
        def _exponential_decay(step, *, lr_init, lr_final):
            t = np.clip(step / self.optim_cfg.max_steps, 0.0, 1.0)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init

        lr_dict = asdict(self.lr_cfg)
        optimizers = {}
        schedulers = {}
        # named parameters will be [part].params.[field]
        # e.g. fg.params.means
        # lr config is a nested dict for each fg/bg part
        for name, params in self.model.named_parameters():
            part, _, field = name.split(".")
            lr = lr_dict[part][field]
            optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])

            if "scales" in name:
                fnc = functools.partial(_exponential_decay, lr_final=0.1 * lr)
            else:
                fnc = lambda _, **__: 1.0

            optimizers[name] = optim
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optim, functools.partial(fnc, lr_init=lr)
            )
        return optimizers, schedulers


def dup_in_optim(optimizer, new_params: list, should_dup: torch.Tensor, num_dups: int):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            p = param_state[key]
            param_state[key] = torch.cat(
                [p[~should_dup], p.new_zeros(num_dups, *p.shape[1:])],
                dim=0,
            )
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def remove_from_optim(optimizer, new_params: list, _should_cull: torch.Tensor):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            param_state[key] = param_state[key][~_should_cull]
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def reset_in_optim(optimizer, new_params: list):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            param_state[key] = torch.zeros_like(param_state[key])
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()
