"""Explicit body-frame geometry and the source implementation's spring model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    return value


def points_array(points):
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Expected finite coordinates with shape (N, 3)")
    return points


@dataclass(frozen=True)
class RobotGeometry:
    vertices: np.ndarray
    edges: np.ndarray
    spring_attachments: np.ndarray
    cage_attachments: np.ndarray
    motor_attachments: np.ndarray
    actuator_names: tuple
    actuator_source_indices: tuple
    actuated_edge_indices: tuple
    sampling_faces: np.ndarray
    outer_radius: float
    core_radius: float
    spring_stiffness: float
    spring_initial_elongation: float
    spring_maximum_elongation: float

    @classmethod
    def from_yaml(cls, path=CONFIG_DIR / "robot.yaml"):
        config = load_yaml(path)
        if (config["schema_version"] != 1 or config["coordinate_frame"] != "body"
                or config["length_unit"] != "m"):
            raise ValueError("Expected schema 1, body coordinates, and metre units")
        records = {}
        for key, count in (("vertices", 12), ("edges", 30), ("springs", 12), ("actuators", 6)):
            rows = config[key]
            names = [row["name"] for row in rows]
            if len(rows) != count or len(set(names)) != count:
                raise ValueError(f"{key}: expected {count} uniquely named entries")
            if any(not name or "*" in name for name in names):
                raise ValueError(f"{key}: names must be explicit")
            records[key] = {row["name"]: i for i, row in enumerate(rows)}
        vertices = points_array([r["position_m"] for r in config["vertices"]])
        edge_pairs = np.array([
            [records["vertices"][name] for name in row["vertices"]]
            for row in config["edges"]
        ])
        if edge_pairs.shape != (30, 2) or len({tuple(sorted(p)) for p in edge_pairs}) != 30:
            raise ValueError("Expected 30 distinct cage edges")
        lengths = np.linalg.norm(vertices[edge_pairs[:, 0]] - vertices[edge_pairs[:, 1]], axis=1)
        if not np.allclose(lengths, config["rod_length_m"], atol=1e-10, rtol=0):
            raise ValueError("Cage edge lengths disagree with rod_length_m")
        if not np.allclose(np.linalg.norm(vertices, axis=1), config["outer_radius_m"], atol=1e-10):
            raise ValueError("Cage vertices disagree with outer_radius_m")
        edges = vertices[edge_pairs].mean(axis=1)
        spring_vertices = [records["vertices"][r["cage_vertex"]] for r in config["springs"]]
        if spring_vertices != list(range(12)):
            raise ValueError("Spring order must match the explicit cage vertex order")
        indices = tuple(r["source_index"] for r in config["actuators"])
        if indices != tuple(range(6)):
            raise ValueError("Actuator order must match source indices 0 through 5")
        active_edges = tuple(records["edges"][r["cage_edge"]] for r in config["actuators"])
        if len(set(active_edges)) != 6:
            raise ValueError("Each actuator must map to a distinct cage edge")
        faces = np.array([[records["actuators"][name] for name in face]
                          for face in config["sampling_faces"]])
        if faces.shape != (8, 3) or any(len(set(face)) != 3 for face in faces):
            raise ValueError("Expected eight triangular sampling faces")
        scalars = [config[k] for k in ("outer_radius_m", "core_radius_m", "spring_stiffness_n_per_m",
                                       "spring_initial_elongation_m", "spring_maximum_elongation_m")]
        if not np.all(np.isfinite(scalars)) or any(x <= 0 for x in scalars):
            raise ValueError("Geometry and spring constants must be finite and positive")
        return cls(
            vertices, edges,
            points_array([r["core_attachment_m"] for r in config["springs"]]),
            edges[list(active_edges)],
            points_array([r["core_attachment_m"] for r in config["actuators"]]),
            tuple(records["actuators"]), indices, active_edges, faces, *scalars,
        )

    def spring_force(self, positions):
        """Source force law: spring directions and lengths measured from core centre."""
        positions = points_array(positions)
        vectors = self.vertices[None, :, :] - positions[:, None, :]
        lengths = np.linalg.norm(vectors, axis=2)
        if np.any(lengths < 1e-12):
            raise ValueError("Spring-force model is singular at a cage vertex")
        elongation = np.maximum(0, lengths - np.linalg.norm(self.vertices, axis=1)
                                + self.spring_initial_elongation)
        return np.sum(self.spring_stiffness * elongation[..., None] * vectors / lengths[..., None], axis=1)

    def spring_elongations(self, positions):
        """Source extension constraint: lengths measured at the CAD attachment points."""
        positions = points_array(positions)
        vectors = positions[:, None, :] + self.spring_attachments - self.vertices
        rest = np.linalg.norm(self.vertices - self.spring_attachments, axis=1)
        return np.maximum(0, np.linalg.norm(vectors, axis=2) - rest + self.spring_initial_elongation)
