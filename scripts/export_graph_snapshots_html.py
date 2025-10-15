#!/usr/bin/env python3
"""Export saved K-NN graph snapshots to an interactive HTML viewer.

The resulting HTML bundles multiple snapshots with a slider and play controls,
so you can download it from a remote machine and open it locally in a browser.
"""

from __future__ import annotations

import argparse
import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import torch


GRAPH_FILENAME_RE = re.compile(r"graph_step_(\d+)\.pt$")


@dataclass
class GraphSnapshot:
    step: int
    means: np.ndarray  # (N, 3)
    segments: np.ndarray  # (E, 2, 3)
    edge_weights: np.ndarray | None  # (E,) distances or None

    @property
    def num_nodes(self) -> int:
        return int(self.means.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.segments.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export graph_step_*.pt snapshots to an interactive HTML viewer."
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("OUTPUT_visdebug/v4_demo_00/graphs"),
        help="Directory containing graph_step_*.pt files.",
    )
    parser.add_argument(
        "--output",
        "--save-path",
        type=Path,
        default=Path("graph_snapshots.html"),
        help="Path to the generated HTML file.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=50,
        help="Load every Nth snapshot to keep the HTML lightweight.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=0,
        help="Maximum number of edges to include per snapshot (<=0 keeps all edges).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for edge subsampling randomness.",
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=3,
        help="Marker size for node points.",
    )
    parser.add_argument(
        "--node-color",
        type=str,
        default="#1f77b4",
        help="Color for nodes in the plotly figure.",
    )
    parser.add_argument(
        "--edge-color",
        type=str,
        default="#ff7f0e",
        help="Color for edges in the plotly figure.",
    )
    parser.add_argument(
        "--node-opacity",
        type=float,
        default=0.6,
        help="Opacity for node markers (0 transparent, 1 opaque).",
    )
    parser.add_argument(
        "--edge-width-min",
        type=float,
        default=0.6,
        help="Minimum edge width after weighting.",
    )
    parser.add_argument(
        "--edge-width-max",
        type=float,
        default=4.0,
        help="Maximum edge width after weighting.",
    )
    parser.add_argument(
        "--edge-width-bins",
        type=int,
        default=5,
        help="Number of width bins; higher values approximate per-edge widths.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=5000,
        help="Downsample to at most this many nodes per snapshot (<=0 keeps all nodes).",
    )
    parser.add_argument(
        "--node-seed",
        type=int,
        default=2025,
        help="Seed for node downsampling when --max-nodes is active.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Optional hard cap on the number of snapshots to load.",
    )
    parser.add_argument(
        "--include-plotlyjs",
        type=str,
        default="embed",
        choices=("embed", "cdn"),
        help="How to include plotly.js. Use 'embed' to view fully offline.",
    )
    return parser.parse_args()


def discover_graph_files(graphs_dir: Path) -> List[Path]:
    pattern = str(graphs_dir / "graph_step_*.pt")
    files = [Path(p) for p in glob.glob(pattern)]
    files.sort(key=lambda p: extract_step(p))
    return files


def extract_step(path: Path) -> int:
    match = GRAPH_FILENAME_RE.search(path.name)
    if match:
        return int(match.group(1))
    return -1


def torch_tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def build_segments(
    means: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray | None,
    max_edges: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Construct a deduplicated list of edge segments for visualization."""
    num_nodes = means.shape[0]
    pairs: List[Tuple[int, int]] = []
    weights: List[float] = []
    seen: Dict[Tuple[int, int], int] = {}

    for src_idx in range(indices.shape[0]):
        for neigh_idx, dst_raw in enumerate(indices[src_idx]):
            dst = int(dst_raw)
            if dst < 0 or dst >= num_nodes or dst == src_idx:
                continue
            edge = (src_idx, dst) if src_idx < dst else (dst, src_idx)
            existing = seen.get(edge)
            if existing is not None:
                if distances is not None:
                    weights[existing] = min(
                        weights[existing], float(distances[src_idx, neigh_idx])
                    )
                continue
            seen[edge] = len(pairs)
            pairs.append(edge)
            if distances is not None:
                weights.append(float(distances[src_idx, neigh_idx]))

    if not pairs:
        return (
            np.empty((0, 2, 3), dtype=np.float32),
            None if distances is None else np.empty((0,), dtype=np.float32),
        )

    pairs_np = np.array(pairs, dtype=np.int64)
    weights_np = np.array(weights, dtype=np.float32) if distances is not None else None
    if max_edges > 0 and pairs_np.shape[0] > max_edges:
        rng = np.random.default_rng(seed)
        choice = rng.choice(pairs_np.shape[0], size=max_edges, replace=False)
        pairs_np = pairs_np[choice]
        if weights_np is not None:
            weights_np = weights_np[choice]

    segments = means[pairs_np]  # (E, 2, 3)
    return segments.astype(np.float32, copy=False), weights_np


class SnapshotCache:
    """Lazy loader with caching for graph snapshots."""

    def __init__(
        self,
        files: Sequence[Path],
        max_edges: int,
        seed: int,
        max_nodes: int | None,
        node_seed: int,
    ):
        self._files = list(files)
        self._max_edges = max_edges
        self._seed = seed
        self._max_nodes = max_nodes if (max_nodes is None or max_nodes > 0) else None
        self._node_seed = node_seed
        self._cache: Dict[int, GraphSnapshot] = {}
        self._node_selection_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._files)

    def _select_nodes(self, num_nodes: int) -> np.ndarray | None:
        if self._max_nodes is None or num_nodes <= self._max_nodes:
            return None
        selection = self._node_selection_cache.get(num_nodes)
        if selection is None:
            rng = np.random.default_rng(self._node_seed)
            selection = np.sort(
                rng.choice(num_nodes, size=self._max_nodes, replace=False)
            ).astype(np.int64)
            self._node_selection_cache[num_nodes] = selection
        return selection

    def _downsample_nodes(
        self,
        means: np.ndarray,
        indices: np.ndarray,
        distances: np.ndarray | None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        selection = self._select_nodes(means.shape[0])
        if selection is None:
            return means, indices, distances

        means_ds = means[selection]
        orig_k = indices.shape[1]
        selected_mask = np.zeros(means.shape[0], dtype=bool)
        selected_mask[selection] = True
        index_map = np.full(means.shape[0], -1, dtype=np.int64)
        index_map[selection] = np.arange(selection.shape[0], dtype=np.int64)

        filtered_indices = np.full(
            (selection.shape[0], orig_k), -1, dtype=np.int64
        )
        filtered_distances = (
            np.zeros((selection.shape[0], orig_k), dtype=distances.dtype)
            if distances is not None
            else None
        )

        for new_idx, old_idx in enumerate(selection):
            neighs = indices[old_idx]
            valid = neighs >= 0
            neighs = neighs[valid]
            if distances is not None:
                neigh_dists = distances[old_idx][valid]
            else:
                neigh_dists = None

            if neighs.size == 0:
                continue

            keep_mask = selected_mask[neighs]
            neighs = neighs[keep_mask]
            if neigh_dists is not None:
                neigh_dists = neigh_dists[keep_mask]
            if neighs.size == 0:
                continue

            mapped = index_map[neighs]
            count = mapped.shape[0]
            filtered_indices[new_idx, :count] = mapped
            if filtered_distances is not None and neigh_dists is not None:
                filtered_distances[new_idx, :count] = neigh_dists

        return means_ds, filtered_indices, filtered_distances

    def load(self, index: int) -> GraphSnapshot:
        if index in self._cache:
            return self._cache[index]

        path = self._files[index]
        raw = torch.load(path, map_location="cpu")

        means = torch_tensor_to_numpy(raw["means"])
        indices = torch_tensor_to_numpy(raw["indices"])
        distances = (
            torch_tensor_to_numpy(raw["distances"])
            if "distances" in raw
            else None
        )

        means, indices, distances = self._downsample_nodes(
            means, indices, distances
        )

        segments, weights = build_segments(
            means,
            indices,
            distances,
            max_edges=self._max_edges,
            seed=self._seed + extract_step(path),
        )

        step = int(raw.get("step", extract_step(path)))
        snapshot = GraphSnapshot(
            step=step,
            means=means,
            segments=segments,
            edge_weights=weights,
        )
        self._cache[index] = snapshot
        return snapshot


def segments_to_line_coords(segments: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
    if segments.size == 0:
        return [], [], []
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for start, end in segments:
        xs.extend([float(start[0]), float(end[0]), None])
        ys.extend([float(start[1]), float(end[1]), None])
        zs.extend([float(start[2]), float(end[2]), None])
    return xs, ys, zs


def compute_axis_ranges(points: np.ndarray, pad_scale: float = 0.05) -> Tuple[List[float], List[float], List[float]]:
    if points.size == 0:
        return [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    pad = np.maximum(ranges * pad_scale, 1e-4)
    xmin, ymin, zmin = (mins - pad).tolist()
    xmax, ymax, zmax = (maxs + pad).tolist()
    return [xmin, xmax], [ymin, ymax], [zmin, zmax]


def build_node_trace(snapshot: GraphSnapshot, args: argparse.Namespace) -> go.Scatter3d:
    return go.Scatter3d(
        x=snapshot.means[:, 0],
        y=snapshot.means[:, 1],
        z=snapshot.means[:, 2],
        mode="markers",
        marker=dict(
            size=args.point_size,
            color=args.node_color,
            opacity=np.clip(args.node_opacity, 0.0, 1.0),
        ),
        name="Nodes",
        hoverinfo="skip",
    )


def compute_edge_widths(
    edge_weights: np.ndarray | None, args: argparse.Namespace
) -> np.ndarray | None:
    if edge_weights is None or edge_weights.size == 0:
        return None
    weights = edge_weights.astype(np.float32, copy=False)
    rng = weights.max() - weights.min()
    if rng <= 1e-12:
        normalized = np.ones_like(weights)
    else:
        normalized = 1.0 - (weights - weights.min()) / rng
    widths = args.edge_width_min + normalized * (
        args.edge_width_max - args.edge_width_min
    )
    return widths


def build_edge_traces(
    snapshot: GraphSnapshot, args: argparse.Namespace
) -> List[go.Scatter3d]:
    num_bins = max(int(args.edge_width_bins), 1)
    widths = compute_edge_widths(snapshot.edge_weights, args)

    if widths is None:
        if snapshot.num_edges == 0:
            widths = np.empty((0,), dtype=np.float32)
        else:
            widths = np.full(
                (snapshot.num_edges,),
                (args.edge_width_min + args.edge_width_max) * 0.5,
                dtype=np.float32,
            )

    traces: List[go.Scatter3d] = []
    if num_bins == 1:
        edge_x, edge_y, edge_z = segments_to_line_coords(snapshot.segments)
        width_val = (
            float(widths.mean())
            if widths.size > 0
            else float((args.edge_width_min + args.edge_width_max) * 0.5)
        )
        traces.append(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color=args.edge_color, width=width_val),
                name="Edges",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        return traces

    if widths.size == 0:
        bin_edges = np.linspace(
            args.edge_width_min, args.edge_width_max, num_bins + 1
        )
    else:
        bin_edges = np.linspace(widths.min(), widths.max(), num_bins + 1)
        if np.allclose(bin_edges[0], bin_edges[-1]):
            bin_edges = np.linspace(
                args.edge_width_min, args.edge_width_max, num_bins + 1
            )

    for bin_idx in range(num_bins):
        if widths.size == 0:
            segs = np.empty((0, 2, 3), dtype=np.float32)
            width_val = float(
                (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) * 0.5
            )
        else:
            lower = bin_edges[bin_idx]
            upper = bin_edges[bin_idx + 1]
            if bin_idx == num_bins - 1:
                mask = (widths >= lower) & (widths <= upper)
            else:
                mask = (widths >= lower) & (widths < upper)
            segs = snapshot.segments[mask]
            if mask.any():
                width_val = float(widths[mask].mean())
            else:
                width_val = float((lower + upper) * 0.5)

        edge_x, edge_y, edge_z = segments_to_line_coords(segs)
        traces.append(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color=args.edge_color, width=max(width_val, 0.1)),
                name=f"Edges bin {bin_idx + 1}",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return traces


def build_figure(
    snapshots: Sequence[GraphSnapshot],
    args: argparse.Namespace,
) -> go.Figure:
    first_snapshot = snapshots[0]
    first_nodes = build_node_trace(first_snapshot, args)
    first_edge_traces = build_edge_traces(first_snapshot, args)
    num_edge_traces = len(first_edge_traces)
    x_range, y_range, z_range = compute_axis_ranges(first_snapshot.means)

    frames = []
    slider_steps = []
    for idx, snapshot in enumerate(snapshots):
        nodes_trace = build_node_trace(snapshot, args)
        edge_traces = build_edge_traces(snapshot, args)
        if len(edge_traces) != num_edge_traces:
            if len(edge_traces) < num_edge_traces:
                missing = num_edge_traces - len(edge_traces)
                edge_traces.extend(
                    [
                        go.Scatter3d(
                            x=[],
                            y=[],
                            z=[],
                            mode="lines",
                            line=dict(
                                color=args.edge_color,
                                width=args.edge_width_min,
                            ),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                        for _ in range(missing)
                    ]
                )
            else:
                edge_traces = edge_traces[:num_edge_traces]
        xr, yr, zr = compute_axis_ranges(snapshot.means)
        frame = go.Frame(
            name=str(idx),
            data=[nodes_trace, *edge_traces],
            layout=dict(
                title=dict(
                    text=f"Graph step {snapshot.step} • Nodes {snapshot.num_nodes} • Edges {snapshot.num_edges}"
                ),
                scene=dict(
                    xaxis=dict(range=xr, title="X"),
                    yaxis=dict(range=yr, title="Y"),
                    zaxis=dict(range=zr, title="Z"),
                ),
            ),
        )
        frames.append(frame)
        slider_steps.append(
            dict(
                args=(
                    [str(idx)],
                    dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                        transition=dict(duration=0),
                    ),
                ),
                label=f"{snapshot.step}",
                method="animate",
            )
        )

    layout = go.Layout(
        title=dict(
            text=f"Graph step {first_snapshot.step} • Nodes {first_snapshot.num_nodes} • Edges {first_snapshot.num_edges}"
        ),
        scene=dict(
            xaxis=dict(range=x_range, title="X"),
            yaxis=dict(range=y_range, title="Y"),
            zaxis=dict(range=z_range, title="Z"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=50),
                steps=slider_steps,
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=1.05,
                yanchor="top",
                xanchor="right",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=200, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0), mode="immediate", transition=dict(duration=0))],
                    ),
                ],
            )
        ],
        showlegend=False,
    )

    fig = go.Figure(
        data=[first_nodes, *first_edge_traces],
        layout=layout,
        frames=frames,
    )
    return fig


def main() -> None:
    args = parse_args()

    graph_files = discover_graph_files(args.graphs_dir)
    if args.stride > 1:
        graph_files = graph_files[:: max(args.stride, 1)]
    if args.limit is not None:
        graph_files = graph_files[: args.limit]

    if not graph_files:
        raise SystemExit(
            f"No graph_step_*.pt files found under {args.graphs_dir.resolve()}"
        )

    cache = SnapshotCache(
        files=graph_files,
        max_edges=args.max_edges,
        seed=args.seed,
        max_nodes=args.max_nodes,
        node_seed=args.node_seed,
    )

    snapshots = [cache.load(i) for i in range(len(cache))]

    fig = build_figure(snapshots, args)
    fig.write_html(
        args.output,
        include_plotlyjs=args.include_plotlyjs,
        auto_open=False,
    )

    print(f"Saved interactive viewer to {args.output.resolve()}")
    print(f"Snapshots included: {len(snapshots)}")
    print(f"Steps: {[snapshot.step for snapshot in snapshots]}")


if __name__ == "__main__":
    main()
