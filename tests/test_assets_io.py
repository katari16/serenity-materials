import struct

import numpy as np

from serenity_paper.io import load_cloud, validate_assets, write_ply, write_stl


def test_all_bundled_assets_and_metadata():
    assert len(validate_assets()) == 7


def test_ascii_pcd_fields_and_gravity(tmp_path):
    path = tmp_path / "cloud.pcd"
    path.write_text("# GRAVITY_VECTOR 0 0 -1\nFIELDS intensity z x y\nDATA ascii\n99 3 1 2\n")
    points, frame, gravity = load_cloud(path)
    np.testing.assert_array_equal(points, [[1, 2, 3]])
    np.testing.assert_array_equal(gravity, [0, 0, -1])
    assert frame is None


def test_mesh_exports_have_correct_counts(tmp_path):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    triangles = np.array([[0, 1, 2]])
    write_stl(tmp_path / "mesh.stl", vertices, triangles)
    raw = (tmp_path / "mesh.stl").read_bytes()
    assert len(raw) == 134
    assert struct.unpack_from("<I", raw, 80)[0] == 1
    np.testing.assert_allclose(np.frombuffer(raw, dtype="<f4", count=9, offset=96).reshape(3, 3), vertices)
    write_ply(tmp_path / "mesh.ply", vertices, triangles)
    assert "element vertex 3" in (tmp_path / "mesh.ply").read_text()
    assert "element face 1" in (tmp_path / "mesh.ply").read_text()
