# Serenity: cave planning and core workspace

Standalone Python support code for the Serenity paper. It provides the cave-direction computation and actuator-limited core workspace calculation from the robot implementation, with explicit geometry, recorded input data, reproducible commands, and numerical exports. ROS 2, motor drivers, camera SDKs, Open3D, and a graphical desktop are not required.

The cave planner estimates a **local travel direction** from observed walls. The workspace routine computes a **sampled boundary of feasible core translations under the source model**. It does not simulate the robot or execute motion.

## Install

Use **Python 3.10, 3.11, or 3.12**. Python 3.12 is the reference environment. From this repository:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps --no-build-isolation -e .
```

On Windows, use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1` instead of the first two lines. All remaining commands are the same. Internet access is needed for dependency installation; the computations then run offline. No separate dataset download is needed.

`requirements.txt` pins every runtime dependency; `requirements-dev.txt` also pins the test/build tools. For a runtime-only installation, install `requirements.txt`, followed by `python -m pip install --no-deps .`.

## Reproduce both computations

```sh
python scripts/reproduce.py --full
```

This validates the bundled data, runs the recorded and synthetic cave examples, and computes the workspace with **150 barycentric samples per coordinate and 300 radial samples**, matching the original sampling settings. Results are written to `results/`. For a fast demonstration, omit `--full` to use a 30-by-60 workspace sampling configuration.

Individual runs:

```sh
python scripts/cave_planner.py --config cave_recorded --output results/cave_recorded
python scripts/cave_planner.py --config cave_synthetic --output results/cave_synthetic
python scripts/reachable_workspace.py --config workspace_full --output results/workspace_full
python scripts/validate_assets.py
python -m pytest -q
```

The installed commands `serenity-cave`, `serenity-workspace`, `serenity-reproduce`, and `serenity-validate` have the same arguments. They also work outside the repository directory because input data and configurations are included in the Python package. Running a command again replaces its generated files in the selected output directory.

## Outputs

| Computation | Files |
|---|---|
| Cave direction | `summary.json`, `cave_result.npz`, `slice.csv`, `cave_direction.png`, `cave_direction.pdf` |
| Reachable workspace | `summary.json`, `workspace.npz`, `boundary.csv`, `boundary.ply`, `boundary.stl`, `reachable_workspace.png`, `reachable_workspace.pdf` |
| Each run | `experiment.yaml`, `robot.yaml`, `run.json` with parameters, package versions, and elapsed time |

All positions are in metres and actuator forces are in newtons. The NPZ and CSV exports retain numerical data; PNG/PDF files are ready for inspection. Workspace PLY/STL files contain the triangulated sampled boundary, with outward-facing triangles. `summary.json` reports whether the mesh is closed and whether any sampled rays lacked a feasible point. The `actuator_names` array in `workspace.npz` gives the force-column order.

## Reference results

The bundled recording contains 921,600 points; 2,375 pass the horizontal slice filter. Two retained wall clusters contain 1,502 and 750 points. The body-frame cave direction is approximately `[0.675071, -0.456375, -0.579656]`; its negative describes the same cave axis. A previous direction can select the travel sign.

| Workspace configuration | Unique boundary points | Triangles | Missing rays | Closed mesh |
|---|---:|---:|---:|---|
| `workspace_quick` | 3,366 | 6,728 | 0 | Yes |
| `workspace_full` | 88,806 | 177,608 | 0 | Yes |

On the reference Apple Silicon / Python 3.12.3 run, the workspace took approximately 3 seconds in quick mode and 30 seconds at full resolution, including exports and plotting. Timing depends on the computer. Floating-point values can vary slightly between platforms. Example figures and numerical summaries are in [`example_results/`](example_results/).

## Configuration and custom inputs

The complete assembly-A geometry is in [`serenity_paper/configs/robot.yaml`](serenity_paper/configs/robot.yaml): all cage vertices and edges, twelve named spring attachments, six named actuator attachments, their source-index mappings, and all physical constants used here. Experiment YAML files in the same directory list their input, filter, sampling, and actuation settings explicitly. They do not inherit or patch other experiment configurations. The `robot` field references the separate complete robot model.

To change a setup, edit an experiment YAML or pass your own file with `--config path/to/experiment.yaml`. Paths inside a configuration are resolved relative to that YAML file. Copy `robot.yaml` alongside a custom configuration, or update its `robot` field to a relative path. Run snapshots preserve parameters for inspection; rerun with the original configuration so its relative data paths resolve correctly.

For a custom body-frame point cloud:

```sh
python scripts/cave_planner.py --input my_cloud.npy --frame body --gravity 0 0 -9.81 --output results/custom_cave
```

Supported inputs are NumPy `.npy` arrays of shape `(N, 3)` or `(H, W, 3)`, `.npz` archives with a `points` array, and ASCII `.pcd` files with `x`, `y`, and `z` fields. Coordinates must already share a common robot frame; raw camera clouds need calibration and transformation before use. NPZ files may include `frame` (`body` or `gravity_aligned`) and `gravity_body`; metadata must agree with the command/configuration.

The bundled cloud is **already gravity-aligned**, as saved by the original planner. Its configuration avoids rotating it a second time. The recorded body-frame gravity is still used to return the estimated direction in body coordinates. Use `--previous-direction X Y Z` to orient the result toward a previous heading expressed in the current body frame.

## Python API

```python
import numpy as np
from serenity_paper.cave import cave_direction
from serenity_paper.geometry import RobotGeometry
from serenity_paper.workspace import evaluate_positions

robot = RobotGeometry.from_yaml()
points = np.load("my_cloud.npy").reshape(-1, 3)
result = cave_direction(points, [0, 0, -9.81], outer_radius=robot.outer_radius)
print(result.direction_body)

state = evaluate_positions(robot, [[0.10, 0.08, 0.06]], force_limit=4830.0)
print(state.feasible, state.forces)  # force columns: actuator_0 through actuator_5
```

## Implementation and scope

- [`cave.py`](serenity_paper/cave.py): gravity alignment, horizontal filtering, single-linkage clustering, orthogonal wall fitting, and direction averaging.
- [`workspace.py`](serenity_paper/workspace.py): actuator selection, force balance, spring-extension checks, and vectorized radial sampling.
- [`geometry.py`](serenity_paper/geometry.py): named geometry validation and the source spring-force model.
- [`data/`](serenity_paper/data/): the complete selected recorded cloud and all six historical workspace geometry files from the source data directory.
- [`tests/`](tests/): source-value regression checks, frame handling, vertical walls, force/extension limits, mapping order, mesh exports, and asset integrity.

The original force model uses core-centre-to-vertex spring directions, while its extension constraint uses the CAD spring attachment points. This distinction is retained. The original acceptance test uses absolute actuator force; the supplied configurations preserve it (`tension_only: false`). Setting `tension_only: true` adds a nonnegative-tension constraint and changes the computation. The default force limit is `4200 mA × 1.15 N/mA = 4830 N`, preserving the executable source expression.

The sampling procedure finds the first feasible point along each ray between radial scales 1.0 and 0.60. It does not test every interior position or certify interpolated mesh faces. The historical meshes are included as reference assets; their exact meshing steps and parameter histories are not available, so this repository regenerates its own explicitly defined surface. See [`docs/METHODS.md`](docs/METHODS.md) for the equations and adaptation details and [`docs/DATA.md`](docs/DATA.md) for file provenance.

## Repository distribution

This directory is ready to place in a Git repository. A GitHub Actions workflow runs tests and reproduction commands on Python 3.10–3.12 across Linux, macOS, and Windows. To build a wheel and source archive locally:

```sh
python -m build --no-isolation
```

Both distributions include the configurations and datasets. Source attribution is in [`NOTICE`](NOTICE); the package uses the Apache-2.0 license declared by the original `serenity_control` package.
