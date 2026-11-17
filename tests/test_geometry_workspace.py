import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from serenity_paper.geometry import CONFIG_DIR, RobotGeometry, load_yaml
from serenity_paper.workspace import evaluate_positions, sample_workspace, triangular_grid


@pytest.fixture
def robot():
    return RobotGeometry.from_yaml()


def test_explicit_actuator_and_spring_order(robot):
    assert robot.actuator_names == ("actuator_0", "actuator_1", "actuator_2", "actuator_3", "actuator_4", "actuator_5")
    assert robot.actuator_source_indices == (0, 1, 2, 3, 4, 5)
    assert robot.actuated_edge_indices == (1, 28, 13, 5, 22, 17)
    np.testing.assert_allclose(robot.cage_attachments, robot.edges[[1, 28, 13, 5, 22, 17]])
    np.testing.assert_allclose(robot.spring_elongations([[0, 0, 0]]), 0.002, atol=1e-15)


@pytest.mark.parametrize("group", ["actuators", "springs"])
def test_reordered_mapping_rejected(tmp_path, group):
    config = load_yaml(CONFIG_DIR / "robot.yaml")
    config[group][0], config[group][1] = config[group][1], config[group][0]
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="order"):
        RobotGeometry.from_yaml(path)


def test_force_equations_against_original_values(robot):
    fixture = json.loads((Path(__file__).parent / "reference_values.json").read_text())
    for case in fixture["cases"]:
        position = case["position_m"]
        result = evaluate_positions(robot, position, 4830.)
        np.testing.assert_allclose(robot.spring_force(position)[0], case["spring_force_n"], rtol=1e-12, atol=1e-10)
        np.testing.assert_array_equal(result.active_indices[0], case["actuator_indices"])
        np.testing.assert_allclose(result.forces[0, case["actuator_indices"]], case["actuator_forces_n"], rtol=1e-11)
        assert result.residual[0] < 1e-8


def test_origin_and_force_limit(robot):
    origin = evaluate_positions(robot, [0, 0, 0], 4830.)
    assert origin.feasible[0]
    np.testing.assert_array_equal(origin.forces, 0)
    assert evaluate_positions(robot, [0.10, 0.08, 0.06], 4830.).feasible[0]
    assert not evaluate_positions(robot, [0.10, 0.08, 0.06], 100.).feasible[0]


def test_force_and_extension_constraints_are_independent(robot):
    position = robot.vertices[0] * 0.8
    result = evaluate_positions(robot, position, 1e12)
    assert result.elongations.max() > robot.spring_maximum_elongation
    assert not result.feasible[0]


def test_optional_tension_constraint(robot):
    point = [0.18, -0.06, -0.11]
    assert evaluate_positions(robot, point, 1e6).feasible[0]
    assert not evaluate_positions(robot, point, 1e6, tension_only=True).feasible[0]


def test_triangular_grid_has_expected_topology():
    uv, triangles = triangular_grid(10)
    assert len(uv) == 55
    assert len(triangles) == 81
    assert np.all(uv.sum(axis=1) <= 1 + 1e-15)
    assert triangles.max() < len(uv)


def test_sampled_boundary_satisfies_source_constraints(robot):
    result = sample_workspace(robot, barycentric_samples=12, radial_samples=30,
                              radial_min=0.6, radial_max=1., current_limit_ma=4200.,
                              force_per_current_n_per_ma=1.15)
    assert len(result.points) > 100
    assert np.all(evaluate_positions(robot, result.points, 4830.).feasible)
    assert np.all((result.radial_scales >= .6) & (result.radial_scales <= 1.))
    corners = result.points[result.triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    assert np.all(np.sum(normals * corners.mean(axis=1), axis=1) >= -1e-15)


@pytest.mark.parametrize("point", [[np.nan, 0, 0], [1, 2], [np.inf, 0, 0]])
def test_invalid_positions_rejected(robot, point):
    with pytest.raises(ValueError):
        evaluate_positions(robot, point, 4830.)
