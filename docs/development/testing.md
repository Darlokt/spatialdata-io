# Test reader changes

## Run tests offline

Run pytest from the repository root after
[installing development dependencies](../contributing.md#installing-dev-dependencies).
This selects unit and synthetic integration tests without downloading datasets.

```bash
uv run --no-sync pytest -m "not data"
```

Put isolated tests under `tests/unit/` at paths matching the corresponding
source modules. Put public workflows, synthetic layouts, CLI workflows and
data-backed regressions in `tests/integration/`. The test collection hook
assigns `unit`, `integration`, reader and CLI markers by path, and assigns
`data` to tests using `require_test_dataset`. Add `slow` to expensive tests.
Run tests with shared resources or process-global state serially unless the
runner isolates them.

## Verify a reader change

For example, check the Visium implementation and its matching tests.

```bash
uv run --no-sync ruff check --no-fix src/spatialdata_io/readers/visium tests/unit/readers/visium tests/integration/readers/visium
uv run --no-sync ruff format --check src/spatialdata_io/readers/visium tests/unit/readers/visium tests/integration/readers/visium
uv run --no-sync pyrefly check src/spatialdata_io/readers/visium tests/unit/readers/visium tests/integration/readers/visium
uv run --no-sync pytest -m "visium and not data"
```

Include any modified CLI, shared or test modules in the static checks. For
an API change, also run `tests/unit/cli/test_cli_alignment.py` and the
affected CLI workflows. Pytest includes branch-aware coverage and missing-line
output by default. A rewritten reader requires at least 90% targeted
branch-aware coverage.

Review the [module docstrings](readers.md#write-module-docstrings) alongside
their tests. Trace a representative source artifact to the documented output
and check that the fixtures cover the stated interpretation and failure cases.

Test invalid inputs at the boundary that owns them. A discovery test can
exercise absent, ambiguous, wrong-kind and escaping paths. Schema tests can
cover reordered, duplicate, missing, null, nonfinite and overflowing fields.
Integration tests should verify exact element names, axes, values, dtypes,
feature and channel order, coordinate systems, transforms and linkage.

For raster changes, verify zero decoded pixel tasks during construction and a
bounded partial read from a compressed tiled or stripped fixture. Include
graph serialization and resource closure where the change affects them.
For large point sources, test projection and deferred row validation. For
expression, assert the expected canonical sparse-array storage without
densifying the input, then verify exact expression and independent target
IDs. For changes to annotation storage, exercise H5AD or Zarr round trips
and check values, missingness, ordering and linkage.

Measure performance changes on identical, representative inputs before and
after the change, in the same environment and after verifying numerical
equivalence. Record input dimensions, projection, chunk or batch sizes, wall
time, peak memory, and task count or bytes read where available.
Time reader construction, an off-origin crop, or a full write according to
the workflow the change affects. State which operation was timed and whether
the cache was warm. For a table parser, compare aligned identifiers and CSR
values and indices row by row. Equal totals can hide a row or feature swap.

## Test against a public dataset

Use the [test dataset guide](datasets.md) to find a registered fixture, install
it locally, and run a data-backed test. The guide also covers registering a new
source and verifying the layout from a clean download.

## Run the broader checks

```bash
uv run --no-sync prek run --all-files
uv run --no-sync pytest -n 6
```

Run serially if the selected tests share resources that cannot be isolated.
Use `-m "not data"` where optional datasets are unavailable. To run the
configured Python and dependency matrix locally, use `uvx hatch test --all`.
CI uses stable dependencies on Python 3.12 and 3.14, plus an advisory
pre-release dependency run on Python 3.14.
[Test workflow](https://github.com/scverse/spatialdata-io/blob/main/.github/workflows/test.yaml)
