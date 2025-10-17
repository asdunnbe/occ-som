from dataclasses import dataclass


@dataclass
class FGLRConfig:
    means: float = 1.6e-4
    opacities: float = 1e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2
    motion_coefs: float = 1e-2


@dataclass
class BGLRConfig:
    means: float = 1.6e-4
    opacities: float = 5e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2


@dataclass
class MotionLRConfig:
    rots: float = 1.6e-4
    transls: float = 1.6e-4

@dataclass
class CameraScalesLRConfig:
    camera_scales: float = 1e-4

@dataclass
class CameraPoseLRConfig:
    Rs: float = 1e-3
    ts: float = 1e-3

@dataclass
class SceneLRConfig:
    fg: FGLRConfig
    bg: BGLRConfig
    motion_bases: MotionLRConfig
    camera_poses: CameraPoseLRConfig
    camera_scales: CameraScalesLRConfig


@dataclass
class LossesConfig:
    w_rgb: float = 1.0
    w_depth_reg: float = 0.5
    w_depth_const: float = 0.1
    w_depth_grad: float = 1
    w_track: float = 2.0
    w_mask: float = 1.0
    w_smooth_bases: float = 0.1
    w_smooth_tracks: float = 2.0
    w_scale_var: float = 0.01
    w_z_accel: float = 1.0

    # w_smooth_bases: float = 0.0
    # w_smooth_tracks: float = 0.0
    # w_scale_var: float = 0.0
    # w_z_accel: float = 0.0


@dataclass
class OptimizerConfig:
    max_steps: int = 5000
    ## Adaptive gaussian control
    warmup_steps: int = 200
    control_every: int = 100
    reset_opacity_every_n_controls: int = 30
    stop_control_by_screen_steps: int = 4000
    stop_control_steps: int = 4000
    ### Densify.
    densify_xys_grad_threshold: float = 0.0002
    densify_scale_threshold: float = 0.01
    densify_screen_threshold: float = 0.05
    stop_densify_steps: int = 15000
    ### Cull.
    cull_opacity_threshold: float = 0.1
    cull_scale_threshold: float = 0.5
    cull_screen_threshold: float = 0.15


@dataclass
class SurfaceModuleConfig:
    enabled: bool = False
    mls_neighbors: int = 16
    projection_iters: int = 3
    bandwidth_scale: float = 2.0
    min_bandwidth: float = 1e-3
    graph_k: int = 6
    graph_update_every: int = 200
    snapshot_every: int = 0
    weight_epsilon: float = 1e-5
    covariance_reg: float = 1e-6
    weight_floor: float = 1e-4
    min_neighbor_weight: float = 1e-2
    min_neighbor_count: int = 3
    decimation_radius: float = 0.0
    max_weight_fraction: float = 0.95
