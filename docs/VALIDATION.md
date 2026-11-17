# Local validation

Validated on 17 November 2026 with Python 3.12.3 on Apple Silicon, using the pinned dependencies.

- All 26 automated tests passed.
- The resolved cage vertices, spring attachments, motor attachments, and edge coordinates matched the original `Geometry` implementation to `1e-15 m` absolute tolerance.
- The recorded cave direction matched the original `CaveDirection` computation within the regression-test angular tolerance.
- The original force equations matched the stored regression fixtures for four core translations, including different actuator selections and signed forces.
- Both recorded and synthetic cave examples completed.
- The full workspace evaluated 16,634,154 candidate positions and found a feasible point on every one of its 90,600 face-local rays. After merging shared edges, it exported 88,806 vertices and 177,608 triangles, with no boundary or nonmanifold edges.
- Every exported boundary point passed the configured force and extension checks. The full run's maximum absolute actuator force was approximately 4829.998343 N, below the 4830 N limit; maximum spring elongation was approximately 0.273387070 m, below the 0.28563 m limit.
- A wheel and source distribution were built. The wheel was installed with pinned runtime dependencies in a separate virtual environment. Both cave examples and the quick/full workspace commands were executed outside the source repository using packaged data and configuration files.
- Dependency compatibility checks passed. PNG figures were inspected visually.

The accompanying GitHub Actions workflow has not been run on hosted runners as part of this local delivery. The checks above validate the supplied numerical implementation and examples; they do not constitute an experimental validation of the robot or every result in the paper.
