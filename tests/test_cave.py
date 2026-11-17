import json
from pathlib import Path

import numpy as np
import pytest

from serenity_paper.cave import aligning_rotation, cave_direction, fit_wall, synthetic_corridor
from serenity_paper.geometry import RobotGeometry
from serenity_paper.io import PACKAGE_DIR, load_cloud


@pytest.mark.parametrize("gravity", [[0, 0, -9.81], [0, 0, 9.81], [2, -3, 8]])
def test_alignment_is_proper_rotation(gravity):
    rotation = aligning_rotation(gravity)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    aligned = np.asarray(gravity) @ rotation
    np.testing.assert_allclose(aligned, [0, 0, -np.linalg.norm(gravity)], atol=1e-14)


def test_vertical_wall_fit():
    points = np.column_stack([np.ones(100), np.linspace(-3, 3, 100)])
    center, direction = fit_wall(points)
    np.testing.assert_allclose(center, [1, 0], atol=1e-14)
    assert abs(direction[1]) == pytest.approx(1., abs=1e-8)


def test_corridor_direction_and_sign():
    gravity = [1.5, -2., -9.]
    rotation = aligning_rotation(gravity)
    points = synthetic_corridor(7, 1200, 8., 2.4, .003, gravity)
    robot = RobotGeometry.from_yaml()
    result = cave_direction(points, gravity, outer_radius=robot.outer_radius)
    expected = np.array([1., 0., 0.]) @ rotation.T
    assert abs(result.direction_body @ expected) > 0.99999
    assert abs(result.direction_body @ np.asarray(gravity)) < 1e-12
    reversed_result = cave_direction(points, gravity, outer_radius=robot.outer_radius, previous_direction=-expected)
    assert reversed_result.direction_body @ expected < -0.99999
    assert result.cluster_sizes == [1200, 1200]


def test_recorded_cloud_frames_are_equivalent():
    points, frame, gravity = load_cloud(PACKAGE_DIR / "data/cave/recorded_cloud.npz")
    robot = RobotGeometry.from_yaml()
    aligned = cave_direction(points, gravity, outer_radius=robot.outer_radius, input_frame=frame)
    body = cave_direction(points @ aligning_rotation(gravity).T, gravity, outer_radius=robot.outer_radius)
    np.testing.assert_allclose(aligned.direction_body, body.direction_body, atol=1e-7)
    assert aligned.input_points == 921600
    assert len(aligned.points_aligned) == 2375
    fixture = json.loads((Path(__file__).parent / "reference_cave.json").read_text())
    assert aligned.cluster_sizes == fixture["cluster_sizes"]
    expected = np.array(fixture["direction_body"])
    assert abs(aligned.direction_body @ expected) > 1 - 1e-9


def test_nan_points_are_removed():
    points = synthetic_corridor(1, 600, 6., 2., .001, [0, 0, -1])
    result = cave_direction(np.vstack([points, [np.nan, 0, 0]]), [0, 0, -1], outer_radius=.5)
    assert result.finite_points == 1200


@pytest.mark.parametrize("gravity", [[0, 0, 0], [np.nan, 0, -1], [1, 2]])
def test_invalid_gravity_rejected(gravity):
    with pytest.raises(ValueError):
        aligning_rotation(gravity)


def test_empty_slice_and_degenerate_wall_fail_clearly():
    with pytest.raises(ValueError, match="horizontal slice"):
        cave_direction(np.zeros((10, 3)), [0, 0, -1], outer_radius=.5)
    with pytest.raises(ValueError, match="coincident"):
        fit_wall(np.ones((500, 2)))
