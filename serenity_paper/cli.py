"""Commands shared by the Python scripts and installed console entry points."""

import argparse
from pathlib import Path
import shutil
import time

import numpy as np
import yaml

from . import __version__
from .cave import cave_direction, synthetic_corridor
from .geometry import CONFIG_DIR, RobotGeometry, load_yaml
from .io import environment, load_cloud, validate_assets, write_json, write_ply, write_stl
from .plotting import plot_cave, plot_workspace
from .workspace import sample_workspace


def config_path(value):
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    bundled = CONFIG_DIR / f"{value}.yaml"
    if bundled.is_file():
        return bundled
    raise ValueError(f"Configuration not found: {value}")


def load_experiment(value):
    path = config_path(value)
    config = load_yaml(path)
    if config["schema_version"] != 1:
        raise ValueError("Unsupported experiment schema")
    robot_path = path.parent / config["robot"]
    return path, config, robot_path, RobotGeometry.from_yaml(robot_path)


def output_dir(value):
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_run(output, config, robot_path, elapsed):
    (output / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    shutil.copyfile(robot_path, output / "robot.yaml")
    write_json(output / "run.json", {"package_version": __version__, "elapsed_seconds": elapsed,
                                    "environment": environment()})


def run_cave(config_value, output_value, *, input_path=None, input_frame=None,
             gravity=None, previous_direction=None):
    started = time.perf_counter()
    path, config, robot_path, robot = load_experiment(config_value)
    source = config["input"]
    if input_path is not None:
        source["kind"] = "recorded"
        source["path"] = str(Path(input_path).resolve())
    if input_frame is not None:
        source["frame"] = input_frame
    if gravity is not None:
        source["gravity_body"] = list(gravity)
    if previous_direction is not None:
        config["planner"]["previous_direction"] = list(previous_direction)
    gravity = np.array(source["gravity_body"])
    if source["kind"] == "recorded":
        cloud_path = path.parent / source["path"]
        points, frame_metadata, gravity_metadata = load_cloud(cloud_path)
        if frame_metadata is not None and frame_metadata != source["frame"]:
            raise ValueError("Input frame disagrees with stored point-cloud metadata")
        if gravity_metadata is not None and not np.allclose(gravity, gravity_metadata, rtol=0, atol=1e-10):
            raise ValueError("Gravity disagrees with stored metadata; use the matching --gravity values")
    elif source["kind"] == "synthetic":
        if source["frame"] != "body":
            raise ValueError("Synthetic corridor is generated in the body frame")
        points = synthetic_corridor(source["seed"], source["points_per_wall"], source["length_m"],
                                    source["width_m"], source["noise_std_m"], gravity)
    else:
        raise ValueError("Input kind must be recorded or synthetic")
    result = cave_direction(points, gravity, outer_radius=robot.outer_radius,
                            input_frame=source["frame"], **config["planner"])
    output = output_dir(output_value)
    np.savez_compressed(output / "cave_result.npz", points_aligned=result.points_aligned,
                        labels=result.labels, direction_body=result.direction_body,
                        direction_aligned=result.direction_aligned, wall_centers=result.wall_centers,
                        wall_directions=result.wall_directions)
    np.savetxt(output / "slice.csv", np.column_stack([result.points_aligned, result.labels]),
               delimiter=",", header="x_aligned_m,y_aligned_m,cluster", comments="", fmt=["%.12g", "%.12g", "%d"])
    summary = {"experiment": config["name"], "input_frame": source["frame"],
               "input_points": result.input_points, "finite_points": result.finite_points,
               "slice_points": len(result.points_aligned), "wall_clusters": len(result.cluster_sizes),
               "cluster_sizes": result.cluster_sizes, "direction_body": result.direction_body.tolist(),
               "direction_gravity_aligned": result.direction_aligned.tolist(),
               "heading_sign": "Toward previous direction when supplied; otherwise an arbitrary axis orientation"}
    write_json(output / "summary.json", summary)
    plot_cave(result, output)
    record_run(output, config, robot_path, time.perf_counter() - started)
    print(f"Cave: {len(result.cluster_sizes)} wall clusters; direction {result.direction_body.round(6).tolist()}")
    print(f"Saved {output}")
    return summary


def run_workspace(config_value, output_value):
    started = time.perf_counter()
    _, config, robot_path, robot = load_experiment(config_value)
    def progress(face, total, found, rays):
        print(f"Workspace face {face}/{total}: {found}/{rays} rays found", flush=True)
    result = sample_workspace(robot, **config["sampling"], **config["actuation"], progress=progress)
    force_limit = config["actuation"]["current_limit_ma"] * config["actuation"]["force_per_current_n_per_ma"]
    output = output_dir(output_value)
    np.savez_compressed(output / "workspace.npz", points_m=result.points, triangles=result.triangles,
                        actuator_forces_n=result.forces, actuator_names=np.array(robot.actuator_names),
                        max_spring_elongation_m=result.max_elongations, radial_scales=result.radial_scales)
    header = "x_m,y_m,z_m," + ",".join(f"{name}_force_n" for name in robot.actuator_names)
    header += ",max_spring_elongation_m,radial_scale"
    np.savetxt(output / "boundary.csv", np.column_stack([result.points, result.forces,
               result.max_elongations, result.radial_scales]), delimiter=",", header=header, comments="", fmt="%.12g")
    write_ply(output / "boundary.ply", result.points, result.triangles)
    write_stl(output / "boundary.stl", result.points, result.triangles)
    summary = {"experiment": config["name"], "coordinate_frame": "body", "length_unit": "m",
               "boundary_points": len(result.points), "triangles": len(result.triangles),
               "rays_total_with_face_duplicates": result.rays_total, "rays_without_feasible_sample": result.rays_missing,
               "candidate_evaluations": result.evaluations, "invalid_equilibrium_candidates": result.invalid_equilibria,
               "mesh_closed": result.closed, "boundary_edges": result.boundary_edges,
               "nonmanifold_edges": result.nonmanifold_edges,
               "force_limit_n": force_limit, "maximum_absolute_force_n": float(np.abs(result.forces).max()),
               "spring_elongation_limit_m": robot.spring_maximum_elongation,
               "maximum_spring_elongation_m": float(result.max_elongations.max()),
               "bounds_m": [result.points.min(axis=0).tolist(), result.points.max(axis=0).tolist()],
               "surface_interpretation": "Triangulated first-feasible radial samples; interpolated faces and interior are not certified feasible"}
    write_json(output / "summary.json", summary)
    plot_workspace(result, robot, output, force_limit)
    record_run(output, config, robot_path, time.perf_counter() - started)
    print(f"Workspace: {len(result.points):,} boundary points; closed mesh: {result.closed}")
    print(f"Saved {output}")
    return summary


def execute(parser, function):
    try:
        return function()
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


def cave_main(argv=None):
    parser = argparse.ArgumentParser(description="Estimate a local cave direction from point-cloud walls")
    parser.add_argument("--config", default="cave_recorded", help="Bundled configuration name or YAML path")
    parser.add_argument("--output", default="results/cave_recorded")
    parser.add_argument("--input", help="Override point cloud: .npz, .npy, or ASCII .pcd")
    parser.add_argument("--frame", choices=["body", "gravity_aligned"])
    parser.add_argument("--gravity", type=float, nargs=3, metavar=("GX", "GY", "GZ"))
    parser.add_argument("--previous-direction", type=float, nargs=3, metavar=("X", "Y", "Z"))
    args = parser.parse_args(argv)
    execute(parser, lambda: run_cave(args.config, args.output, input_path=args.input, input_frame=args.frame,
                                    gravity=args.gravity, previous_direction=args.previous_direction))


def workspace_main(argv=None):
    parser = argparse.ArgumentParser(description="Compute the actuator-limited core workspace boundary")
    parser.add_argument("--config", default="workspace_quick", help="Bundled configuration name or YAML path")
    parser.add_argument("--output", default="results/workspace_quick")
    args = parser.parse_args(argv)
    execute(parser, lambda: run_workspace(args.config, args.output))


def validate_main(argv=None):
    parser = argparse.ArgumentParser(description="Validate bundled assets and explicit actuator/spring mappings")
    parser.parse_args(argv)
    def validate():
        geometry = RobotGeometry.from_yaml()
        assets = validate_assets()
        print(f"Validated {len(assets)} bundled assets, {len(geometry.actuator_names)} actuators, and 12 springs")
    execute(parser, validate)


def reproduce_main(argv=None):
    parser = argparse.ArgumentParser(description="Run both cave examples and the core workspace computation")
    parser.add_argument("--full", action="store_true", help="Use the original 150 x 150 barycentric / 300 radial resolution")
    parser.add_argument("--output", default="results")
    args = parser.parse_args(argv)
    def reproduce():
        validate_assets()
        output = Path(args.output)
        summaries = {}
        for name in ("cave_recorded", "cave_synthetic"):
            summaries[name] = run_cave(name, output / name)
        name = "workspace_full" if args.full else "workspace_quick"
        summaries[name] = run_workspace(name, output / name)
        write_json(output / "reproduction_summary.json", summaries)
    execute(parser, reproduce)
