"""Portable input, numerical exports, and bundled-asset validation."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import struct

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent


def load_cloud(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            points = data["points"].copy()
            frame = str(data["frame"].item()) if "frame" in data else None
            gravity = data["gravity_body"].copy() if "gravity_body" in data else None
        return points, frame, gravity
    if path.suffix == ".npy":
        points = np.load(path, allow_pickle=False)
        if points.ndim not in (2, 3) or points.shape[-1] != 3:
            raise ValueError("NPY cloud must have shape (N, 3) or (H, W, 3)")
        return points.reshape(-1, 3), None, None
    if path.suffix == ".pcd":
        gravity = None
        with path.open(encoding="utf-8") as stream:
            fields = None
            for line in stream:
                if line.startswith("# GRAVITY_VECTOR"):
                    gravity = np.array([float(v) for v in line.split()[2:]])
                if line.startswith("FIELDS"):
                    fields = line.split()[1:]
                if line.startswith("DATA"):
                    if line.strip() != "DATA ascii":
                        raise ValueError("Only ASCII PCD files are supported")
                    break
            else:
                raise ValueError("Missing PCD DATA header")
            if fields is None or not all(axis in fields for axis in ("x", "y", "z")):
                raise ValueError("PCD must contain x, y, z fields")
            points = np.loadtxt(stream, usecols=[fields.index(axis) for axis in ("x", "y", "z")], ndmin=2)
        return points, None, gravity
    raise ValueError("Expected an .npz, .npy, or ASCII .pcd point cloud")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def environment():
    return {"python": platform.python_version(), "platform": platform.system(),
            "architecture": platform.machine(),
            "dependencies": {name: importlib.metadata.version(name)
                             for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "PyYAML")}}


def write_ply(path, points, triangles):
    with Path(path).open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\n")
        stream.write(f"element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n")
        np.savetxt(stream, points, fmt="%.12g")
        np.savetxt(stream, np.column_stack([np.full(len(triangles), 3), triangles]), fmt="%d")


def write_stl(path, points, triangles):
    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-30)
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    records = np.zeros(len(triangles), dtype=dtype)
    records["normal"], records["vertices"] = normals, corners
    with Path(path).open("wb") as stream:
        stream.write(b"Serenity sampled workspace boundary; coordinates in metres".ljust(80, b" "))
        stream.write(struct.pack("<I", len(triangles)))
        stream.write(records.tobytes())


def validate_assets():
    manifest = json.loads((PACKAGE_DIR / "data/manifest.json").read_text(encoding="utf-8"))
    checked = []
    for item in manifest["assets"]:
        path = PACKAGE_DIR / item["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Asset missing or modified: {item['path']}")
        if path.suffix == ".stl":
            with path.open("rb") as stream:
                stream.seek(80)
                triangles = struct.unpack("<I", stream.read(4))[0]
            if path.stat().st_size != 84 + 50 * triangles:
                raise ValueError(f"Invalid binary STL length: {path.name}")
        checked.append(item["path"])
    points, frame, gravity = load_cloud(PACKAGE_DIR / "data/cave/recorded_cloud.npz")
    if points.shape != (921600, 3) or frame != "gravity_aligned" or gravity.shape != (3,):
        raise ValueError("Recorded cloud metadata or dimensions changed")
    return checked
