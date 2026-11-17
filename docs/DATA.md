# Bundled data

All computation inputs are included below `serenity_paper/`; normal execution never reads the original ROS workspace. The data are also included in built wheels and source distributions.

## Recorded cave cloud

`data/cave/recorded_cloud.npz` contains all 921,600 points from the source `data/latest_cave_cloud.pcd`, without filtering, subsampling, or coordinate rounding. ASCII coordinates were parsed into NumPy float64 values and compressed losslessly. The archive contains:

- `points`: `(921600, 3)` float64 coordinates, already gravity-aligned by the original planner.
- `gravity_body`: `[-3.707652451302842, -8.710494434895647, 2.540000000000002]`, copied from the PCD header.
- `frame`: the string `gravity_aligned`.

The body's gravity vector and the stored coordinates are expressed in different frames by design, reflecting the original save function. The recorded experiment configuration specifies this distinction. The synthetic corridor is generated deterministically from the seed and dimensions in `cave_synthetic.yaml`; it is an algorithm check rather than an experimental recording.

## Historical workspace geometry

The complete set of six workspace geometry files present in the source package's data directory is copied byte for byte into `data/reference_workspaces/`:

- `4200mA_limit_with_spring _max_elong.ply`
- `4200mA_limit_with_spring _max_elong.stl`
- `4200mA_not_watertight.stl`
- `core_positions_0.56m_4.6kN.stl`
- `core_positions_4000N_spring_elong.stl`
- `core_positions_high_res.stl`

Names are preserved, including the space before `_max_elong`. These assets are available for visual comparison. The workspace calculation uses the explicit geometry and constraints in its YAML files rather than loading these meshes. Different names indicate historical variants; their exact parameter settings and mesh-generation histories are not inferred from filenames. In particular, this package does not claim that its generated STL is byte-identical to any historical STL.

## Integrity and source identification

`data/manifest.json` records the original repository revision, source-relative filenames, and SHA-256 hashes of all seven bundled assets. The recording entry also includes the original PCD's hash and the lossless conversion description. Run `python scripts/validate_assets.py` to verify hashes, binary STL sizes, cloud dimensions/frame metadata, and geometry mappings.

Original images, camera SDK libraries, motor drivers, and unrelated recordings are not required by either numerical workflow. They are not distributed here. The two workflows run from the bundled input and explicit configuration alone.
