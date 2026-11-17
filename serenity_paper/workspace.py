"""Actuator equilibrium and radial sampling of the core workspace boundary."""

from dataclasses import dataclass

import numpy as np

from .geometry import points_array


@dataclass
class Feasibility:
    feasible: np.ndarray
    forces: np.ndarray
    elongations: np.ndarray
    valid_equilibrium: np.ndarray
    active_indices: np.ndarray
    residual: np.ndarray


def evaluate_positions(geometry, positions, force_limit, *, tension_only=False):
    """Solve the source three-actuator selection and check force/extension limits."""
    positions = points_array(positions)
    if not np.isfinite(force_limit) or force_limit <= 0:
        raise ValueError("force_limit must be positive and finite")
    n = len(positions)
    vectors = geometry.cage_attachments - geometry.motor_attachments - positions[:, None, :]
    lengths = np.linalg.norm(vectors, axis=2)
    unit = vectors / np.maximum(lengths[..., None], 1e-15)
    active = positions @ geometry.cage_attachments.T > 0
    ids = np.argsort(~active, axis=1, kind="stable")[:, :3]
    matrices = np.transpose(np.take_along_axis(unit, ids[..., None], axis=1), (0, 2, 1))
    valid = ((active.sum(axis=1) == 3) & np.all(lengths > 1e-12, axis=1)
             & (np.abs(np.linalg.det(matrices)) > 1e-12))
    forces = np.zeros((n, 6))
    spring = geometry.spring_force(positions)
    selected = np.full((n, 3), np.nan)
    selected[valid] = np.linalg.solve(matrices[valid], -spring[valid, :, None])[..., 0]
    rows = np.flatnonzero(valid)
    forces[rows[:, None], ids[valid]] = selected[valid]
    centered = np.linalg.norm(positions, axis=1) < 1e-14
    valid[centered] = np.linalg.norm(spring[centered], axis=1) < 1e-8
    ids[centered] = -1
    residual = np.linalg.norm(np.sum(forces[..., None] * unit, axis=1) + spring, axis=1)
    valid &= residual < 1e-7
    elongations = geometry.spring_elongations(positions)
    feasible = (valid & (np.max(np.abs(forces), axis=1) <= force_limit)
                & (np.max(elongations, axis=1) <= geometry.spring_maximum_elongation))
    if tension_only:
        feasible &= np.all(forces >= -1e-9, axis=1)
    forces[~valid] = np.nan
    ids[~valid] = -1
    return Feasibility(feasible, forces, elongations, valid, ids, residual)


def triangular_grid(samples):
    if not isinstance(samples, int) or samples < 2:
        raise ValueError("barycentric_samples must be an integer >= 2")
    pairs = [(i, j) for i in range(samples) for j in range(samples - i)]
    index = {pair: k for k, pair in enumerate(pairs)}
    triangles = []
    for i, j in pairs:
        if i + j < samples - 1:
            triangles.append([index[i, j], index[i + 1, j], index[i, j + 1]])
        if i + j < samples - 2:
            triangles.append([index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]])
    return np.asarray(pairs, dtype=float) / (samples - 1), np.asarray(triangles)


@dataclass
class WorkspaceResult:
    points: np.ndarray
    triangles: np.ndarray
    forces: np.ndarray
    max_elongations: np.ndarray
    radial_scales: np.ndarray
    rays_total: int
    rays_missing: int
    evaluations: int
    invalid_equilibria: int
    boundary_edges: int
    nonmanifold_edges: int
    closed: bool


def sample_workspace(geometry, *, barycentric_samples, radial_samples,
                     radial_min, radial_max, current_limit_ma, force_per_current_n_per_ma,
                     tension_only=False, progress=None):
    """Keep the first feasible sample on each ray, scanning from outer to inner."""
    if (not isinstance(radial_samples, int) or radial_samples < 2
            or not 0 < radial_min < radial_max <= 1
            or not np.isfinite(current_limit_ma) or current_limit_ma <= 0
            or not np.isfinite(force_per_current_n_per_ma) or force_per_current_n_per_ma <= 0):
        raise ValueError("Invalid radial sampling or actuator-limit configuration")
    uv, local_triangles = triangular_grid(barycentric_samples)
    force_limit = current_limit_ma * force_per_current_n_per_ma
    anchors = geometry.cage_attachments
    radii = np.linalg.norm(anchors, axis=1)
    if np.any(radii <= geometry.core_radius):
        raise ValueError("Core radius must be smaller than cage attachment radii")
    outer = anchors * (1 - geometry.core_radius / radii[:, None])
    all_points, all_scales, all_triangles = [], [], []
    missing = evaluations = invalid = 0
    offset = 0
    for face_index, face in enumerate(geometry.sampling_faces):
        p1, p2, p3 = outer[face]
        rays = p1 + uv[:, :1] * (p2 - p1) + uv[:, 1:] * (p3 - p1)
        found = np.zeros(len(rays), dtype=bool)
        surface = np.full_like(rays, np.nan)
        scales = np.full(len(rays), np.nan)
        for scale in np.linspace(radial_max, radial_min, radial_samples):
            pending = np.flatnonzero(~found)
            if not len(pending):
                break
            candidates = rays[pending] * scale
            state = evaluate_positions(geometry, candidates, force_limit, tension_only=tension_only)
            evaluations += len(candidates)
            invalid += int((~state.valid_equilibrium).sum())
            accepted = pending[state.feasible]
            surface[accepted] = candidates[state.feasible]
            scales[accepted] = scale
            found[accepted] = True
        remap = np.full(len(rays), -1, dtype=int)
        remap[found] = np.arange(found.sum()) + offset
        triangles = remap[local_triangles]
        triangles = triangles[np.all(triangles >= 0, axis=1)]
        all_points.append(surface[found])
        all_scales.append(scales[found])
        all_triangles.append(triangles)
        missing += int((~found).sum())
        offset += int(found.sum())
        if progress:
            progress(face_index + 1, len(geometry.sampling_faces), int(found.sum()), len(rays))
    if offset < 4:
        raise ValueError("Too few feasible samples; check limits and radial search range")
    points = np.vstack(all_points)
    scales = np.concatenate(all_scales)
    triangles = np.vstack(all_triangles)
    _, first, remap = np.unique(np.round(points, 12), axis=0, return_index=True, return_inverse=True)
    points, scales = points[first], scales[first]
    triangles = remap[triangles]
    distinct = ((triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
                & (triangles[:, 0] != triangles[:, 2]))
    triangles = triangles[distinct]
    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inward = np.sum(normals * corners.mean(axis=1), axis=1) < 0
    triangles[inward] = triangles[inward, ::-1]
    edge_pairs = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    _, counts = np.unique(np.sort(edge_pairs, axis=1), axis=0, return_counts=True)
    boundary = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))
    state = evaluate_positions(geometry, points, force_limit, tension_only=tension_only)
    if not np.all(state.feasible):
        raise RuntimeError("Exported boundary contains infeasible samples")
    return WorkspaceResult(points, triangles, state.forces, state.elongations.max(axis=1), scales,
                           len(uv) * len(geometry.sampling_faces), missing, evaluations, invalid,
                           boundary, nonmanifold, bool(len(triangles) and boundary == 0 and nonmanifold == 0))
