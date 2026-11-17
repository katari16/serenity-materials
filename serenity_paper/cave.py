"""Gravity alignment, single-linkage wall clustering, and orthogonal line fitting."""

from dataclasses import dataclass

import numpy as np
from scipy import odr
from sklearn.cluster import AgglomerativeClustering


def aligning_rotation(gravity):
    """Return R such that row-vector body points transform as points @ R."""
    gravity = np.asarray(gravity, dtype=float)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)) or np.linalg.norm(gravity) < 1e-12:
        raise ValueError("Gravity must be a finite nonzero three-vector")
    target = -gravity / np.linalg.norm(gravity)
    cosine = float(np.clip(target[2], -1, 1))
    if cosine > 1 - 1e-14:
        return np.eye(3)
    if cosine < -1 + 1e-14:
        return np.diag([1., -1., -1.])
    axis = np.cross([0., 0., 1.], target)
    sine = np.linalg.norm(axis)
    axis /= sine
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + sine * skew + (1 - cosine) * (skew @ skew)


def fit_wall(points):
    """Fit an isotropic ODR line, swapping axes for near-vertical walls."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError("Wall points must be a finite array with shape (N, 2), N >= 2")
    center = points.mean(axis=0)
    variances = np.var(points, axis=0)
    if variances.max() < 1e-18:
        raise ValueError("Cannot fit a wall to coincident points")
    independent = int(np.argmax(variances))
    dependent = 1 - independent
    if variances[dependent] < 1e-18:
        return center, np.eye(2)[independent]
    x, y = points[:, independent], points[:, dependent]
    initial = np.polyfit(x, y, 1)[::-1]
    model = odr.Model(lambda beta, x: beta[0] + beta[1] * x)
    result = odr.ODR(odr.Data(x, y, wd=1., we=1.), model, beta0=initial).run()
    if result.info not in (1, 2, 3) or not np.all(np.isfinite(result.beta)):
        raise ValueError(f"Wall regression failed: {result.stopreason}")
    direction = np.empty(2)
    direction[independent], direction[dependent] = 1., result.beta[1]
    direction /= np.linalg.norm(direction)
    if direction[0] < 0 or (abs(direction[0]) < 1e-12 and direction[1] < 0):
        direction *= -1
    return center, direction


@dataclass
class CaveResult:
    direction_body: np.ndarray
    direction_aligned: np.ndarray
    points_aligned: np.ndarray
    labels: np.ndarray
    wall_centers: np.ndarray
    wall_directions: np.ndarray
    cluster_sizes: list
    input_points: int
    finite_points: int


def cave_direction(points, gravity, *, outer_radius, clearance=0.20,
                   slice_half_height=0.05, distance_threshold=0.05,
                   min_cluster_points=500, input_frame="body", previous_direction=None):
    """Estimate a local cave axis; optionally orient it toward a previous body-frame heading."""
    if (not np.all(np.isfinite([outer_radius, clearance, slice_half_height, distance_threshold]))
            or min(outer_radius, slice_half_height, distance_threshold) <= 0 or clearance < 0
            or not isinstance(min_cluster_points, (int, np.integer)) or min_cluster_points < 2):
        raise ValueError("Invalid cave filtering or clustering parameters")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3)")
    count = len(points)
    points = points[np.all(np.isfinite(points), axis=1)]
    finite_count = len(points)
    rotation = aligning_rotation(gravity)
    if input_frame == "body":
        aligned = points @ rotation
    elif input_frame == "gravity_aligned":
        aligned = points
    else:
        raise ValueError("input_frame must be body or gravity_aligned")
    mask = ((np.linalg.norm(aligned, axis=1) >= outer_radius + clearance)
            & (np.abs(aligned[:, 2]) <= slice_half_height))
    xy = aligned[mask, :2]
    if len(xy) < min_cluster_points:
        raise ValueError(f"Only {len(xy)} points in the horizontal slice; need {min_cluster_points}")
    labels = AgglomerativeClustering(n_clusters=None, linkage="single",
                                    distance_threshold=distance_threshold).fit_predict(xy)
    unique, counts = np.unique(labels, return_counts=True)
    large = unique[counts >= min_cluster_points]
    if len(large) == 0:
        raise ValueError("No wall cluster reaches min_cluster_points")
    centers, directions, sizes = [], [], []
    for label in large:
        wall = xy[labels == label]
        center, direction = fit_wall(wall)
        centers.append(center)
        directions.append(direction)
        sizes.append(len(wall))
    directions = np.asarray(directions)
    directions[np.dot(directions, directions[0]) < 0] *= -1
    heading = directions.sum(axis=0)
    heading /= np.linalg.norm(heading)
    aligned_heading = np.r_[heading, 0.]
    body_heading = aligned_heading @ rotation.T
    if previous_direction is not None:
        previous = np.asarray(previous_direction, dtype=float)
        if previous.shape != (3,) or not np.all(np.isfinite(previous)) or np.linalg.norm(previous) < 1e-12:
            raise ValueError("Previous direction must be a finite nonzero three-vector")
        if body_heading @ previous < 0:
            body_heading *= -1
            aligned_heading *= -1
    return CaveResult(body_heading, aligned_heading, xy, np.where(np.isin(labels, large), labels, -1),
                      np.asarray(centers), directions, sizes, count, finite_count)


def synthetic_corridor(seed, points_per_wall, length, width, noise_std, gravity):
    """Create a deterministic two-wall fixture with known horizontal direction."""
    if (not isinstance(points_per_wall, int) or points_per_wall < 2 or length <= 0
            or width <= 0 or noise_std < 0):
        raise ValueError("Invalid synthetic corridor parameters")
    rng = np.random.default_rng(seed)
    x = np.linspace(-length / 2, length / 2, points_per_wall)
    walls = [np.column_stack([x, np.full_like(x, sign * width / 2), np.zeros_like(x)])
             for sign in (-1, 1)]
    points = np.vstack(walls) + rng.normal(0, noise_std, (2 * points_per_wall, 3))
    return points @ aligning_rotation(gravity).T
