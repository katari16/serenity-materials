"""Headless figures for inspection and paper preparation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_cave(result, output):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for i, label in enumerate(np.unique(result.labels)):
        points = result.points_aligned[result.labels == label]
        ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.55,
                   color="#adb5bd" if label == -1 else palette(i % 10),
                   label="Small clusters" if label == -1 else f"Wall cluster ({len(points)} points)",
                   rasterized=True)
    extent = max(float(np.ptp(result.points_aligned, axis=0).max()), 1.)
    for center, direction in zip(result.wall_centers, result.wall_directions):
        segment = center + np.array([[-0.45], [0.45]]) * extent * direction
        ax.plot(segment[:, 0], segment[:, 1], color="#cc4c3b", lw=1.5)
    arrow = result.direction_aligned[:2] * extent * 0.25
    ax.annotate("", xy=arrow, xytext=(0, 0), arrowprops={"color": "#163b66", "width": 2, "headwidth": 10})
    ax.scatter([0], [0], s=40, c="#163b66", marker="x", label="Robot centre")
    ax.set(xlabel="Gravity-aligned x (m)", ylabel="Gravity-aligned y (m)",
           title="Cave direction from a horizontal point-cloud slice", aspect="equal")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    save_figure(fig, output, "cave_direction")


def plot_workspace(result, geometry, output, force_limit):
    fig = plt.figure(figsize=(11, 4.7), constrained_layout=True)
    ax = fig.add_subplot(121, projection="3d")
    xy = fig.add_subplot(122)
    stride = max(1, len(result.points) // 10000)
    points = result.points[::stride]
    load = np.max(np.abs(result.forces[::stride]), axis=1) / force_limit
    cloud = ax.scatter(*points.T, c=load, s=2, cmap="viridis_r", vmin=0, vmax=1, rasterized=True)
    ax.scatter(*geometry.cage_attachments.T, c="#e0783c", s=22, marker="^", label="Cage actuators")
    bound = np.max(np.abs(geometry.cage_attachments)) * 1.08
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-bound, bound),
           xlabel="Body x (m)", ylabel="Body y (m)",
           title=f"Sampled boundary · {len(result.points):,} points")
    ax.text2D(-0.08, 0.5, "Body z (m)", transform=ax.transAxes, rotation=90, va="center")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=22, azim=35)
    ax.legend(fontsize=8, loc="upper left")
    xy.scatter(points[:, 0], points[:, 1], c=load, s=2, cmap="viridis_r", vmin=0, vmax=1, rasterized=True)
    xy.set(xlabel="Body x (m)", ylabel="Body y (m)", aspect="equal", title="Boundary projected onto x–y")
    xy.grid(alpha=0.2)
    fig.colorbar(cloud, ax=[ax, xy], label="Maximum absolute actuator force / limit", shrink=0.78)
    save_figure(fig, output, "reachable_workspace")
