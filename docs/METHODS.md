# Numerical methods

The code is adapted from `src/serenity_control/serenity_control/` at source revision `399535d6dbf1f06014764cd725fdc2e86baafffc`. The original files are `utils/cave_path_planner.py`, `scripts/core_space_movement_gen.py`, `utils/geometry.py`, and `utils/math_utils.py`. The force/current conversion is also used in `control_node.py`.

## Cave-direction estimation

1. Construct a rotation `R` that maps the aligned positive z-axis to the negative normalized body-frame gravity vector. For row-vector input coordinates, `p_aligned = p_body @ R`.
2. Retain finite points whose distance from the robot centre is at least `outer_radius + 0.20 m`, and whose aligned z-coordinate is within `±0.05 m`.
3. Project onto aligned x–y and apply single-linkage agglomerative clustering with a `0.05 m` distance threshold. Retain clusters with at least 500 points.
4. Fit a line to each retained cluster using isotropic orthogonal distance regression (ODR, equal x/y weights). Convert each line to a unit direction, flip inconsistent signs relative to the first direction, and normalize their unweighted sum.
5. Transform the resulting horizontal direction back to body coordinates with `direction_body = direction_aligned @ R.T`.

The original ROS planner writes its cloud **after** applying `points @ R`. The supplied NPZ stores these exact parsed coordinates. Its `gravity_aligned` configuration therefore bypasses step 1 for the points, while retaining the body-frame gravity for step 5. Raw camera arrays in the source workspace were not used as inputs to this example because they require additional sensor transformations.

The estimator returns an axis with a sign ambiguity. The standalone interface optionally chooses the sign by a dot product with a previous heading in the **current** body frame. It does not perform odometry or transform a heading from an earlier robot pose. This is a local direction estimator, not a collision-free global path search or a complete jumping controller.

The implementation retains scikit-learn single-linkage clustering and SciPy ODR. Axes are exchanged when necessary to fit near-vertical walls; exactly axis-aligned walls are handled analytically. Zero/invalid gravity, empty slices, and degenerate clusters produce clear errors. Parallel and antiparallel gravity are handled explicitly. Plot arrows remain in the same frame as the plotted points.

Library references: [AgglomerativeClustering](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.AgglomerativeClustering.html), [SciPy ODR](https://docs.scipy.org/doc/scipy-1.13.1/reference/odr.html).

## Core geometry and force balance

`robot.yaml` lists the complete resolved assembly-A geometry in metres. It contains twelve cage vertices `v_i`, twelve core spring attachments `s_i`, six core motor attachments `m_j`, and thirty explicitly named cage edges. The cage actuator attachment `a_j` is the midpoint of its named edge. Actuator indices 0–5 map to edge indices `[1, 28, 13, 5, 22, 17]`, as in the source. These are array indices, not a claim about electrical bus IDs.

The resolved coordinates preserve the original CAD transformations, spring permutation `[3, 0, 2, 1, 4, 7, 5, 6, 8, 11, 9, 10]`, motor scale `1.3`, and motor permutation `[4, 3, 5, 1, 2, 0]`. No CAD processing, hidden configuration generation, URDF, MJCF, or external asset path is needed. The source package supplied this geometry in Python rather than a robot-description asset folder.

The source spring-force law for a core translation `x` is:

```text
L_i(x) = ||v_i - x||
e_i(x) = max(0, L_i(x) - ||v_i|| + e_0)
F_s(x) = sum_i k * e_i(x) * (v_i - x) / L_i(x)
```

Here `k = 4740 N/m` and `e_0 = 0.002 m`. This force law treats springs as pointing toward the core centre. Separately, the source checks physical spring elongations using the CAD attachment points:

```text
E_i(x) = max(0, ||x + s_i - v_i|| - ||s_i - v_i|| + e_0)
E_i(x) <= 0.28563 m
```

These are deliberately separate equations, matching the original implementation.

Select the three actuator indices with `dot(a_j, x) > 0`. For each selected actuator:

```text
u_j(x) = (a_j - m_j - x) / ||a_j - m_j - x||
A(x) = [u_j0, u_j1, u_j2]
A(x) f = -F_s(x)
```

The standalone code uses a linear solve instead of explicitly computing `inverse(A)`. Invalid selections, singular systems, or force-balance residuals above `1e-7 N` are rejected. The balanced centre has zero actuator forces and is handled directly. Source geometry avoids invalid systems in the supplied full-resolution sampling run.

The original force constraint is `abs(f_j) <= 4200 * 1.15 = 4830 N`. The numerical expression is preserved; an old source comment referred to a different force limit. By default, negative force solutions are allowed because the source also checked absolute values. Set `tension_only: true` to require nonnegative selected tensions as an additional model restriction. The conversion and constraints are explicit in each experiment YAML.

The model covers translational static force balance under those assumptions. It does not add gravity loading, torque balance, core rotation, cable routing/collisions, actuator dynamics, or a search over all possible actuator subsets.

## Boundary sampling and mesh construction

The six cage actuator points define eight explicit sampling faces. Each face corner is shifted radially inward by the `0.09 m` core radius. A triangular barycentric lattice defines one ray per surface point. Radial scale is scanned from 1.0 down to 0.60; the first candidate satisfying the force and spring-extension criteria is retained. The source's early-stop-per-ray procedure is preserved, and all pending candidates at a radial step are evaluated in a NumPy batch.

The full configuration uses 150 samples per barycentric coordinate and 300 radial scales. The triangular domain contains 11,325 rays per face, or 90,600 including shared face edges. Sampling uses integer lattice indices to include every boundary edge consistently, avoiding floating-point `u + v <= 1` omissions. No random sampling is used.

The standalone exporter connects neighbouring accepted lattice points. Points on shared face edges are merged using a 12-decimal-place coordinate key; the original float64 coordinate of one representative is retained. Face normals are oriented outward. Missing rays leave holes rather than being filled by a convex hull. The output records boundary and nonmanifold edge counts. This meshing step is supplied here because the original script exports a point cloud but does not include its historical STL construction procedure.

The output is a sampled boundary. A closed mesh establishes topological closure, not continuous physical feasibility: triangle interiors and the volume inside the surface are not exhaustively checked. Disconnected feasible intervals along a ray are also not enumerated. The code reports sampled points and bounds, and deliberately does not label a convex-hull volume as a proven reachable volume.

## Validation and numerical reproducibility

Tests verify actuator/spring ordering, cage dimensions, zero-deflection spring elongation, numerical forces recorded from the original equations, the original cave direction, rotation handling, exact vertical walls, constraint enforcement, and export/asset consistency. The reference cave direction agrees to the tested angular tolerance after reconstructing body-frame points from the saved aligned cloud. The full run validates every exported boundary point again before writing results.

Pinned dependencies are provided for Python 3.10–3.12. Reference runs were executed with Python 3.12.3 on Apple Silicon. The GitHub workflow is supplied for broader CI execution; those hosted jobs are not represented as already run. Minor numerical/plotting differences may occur across BLAS libraries and operating systems. Each run records the installed direct dependency versions.
