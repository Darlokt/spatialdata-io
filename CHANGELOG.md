# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

Release notes for `v0.7.1` and earlier are available on the [Releases][] page.

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/
[releases]: https://github.com/scverse/spatialdata-io/releases

## [Unreleased]

### Added

- `spatialdata_io` ships a `py.typed` marker, so downstream type checkers use its annotations.
- The generic image reader supports 2D and 3D grayscale or multichannel axes,
  explicit TIFF series and pyramid-level selection, and `.btf` inputs.
- The Xenium reader supports XOA 1.0 to 4.x bundles, Xenium Ranger outputs and
  stripped CSV/MatrixMarket deposits, Protein Expression with its scaling factor,
  cell circles, an explicit `table_target`, the shared `invalid_geometries`
  policy, opt-in secondary analysis and run metadata in `attrs["xenium"]`. See
  the Xenium format guide.

### Changed

- **Breaking:** `to_legacy_anndata()` preserves table positions by default and
  offers explicit `position_source="centroids"` with ID-aligned linked targets,
  one selected coordinate frame and preservation of replaced physical positions.
  `from_legacy_anndata()` imports plain legacy coordinates as pixels and creates
  linked Points while preserving original identifiers. Multiple legacy libraries
  require explicit library/observation association; consumed image scales must
  be positive and finite. Both functions return owned results with CSR-array
  expression. See the AnnData access and legacy conversion guide.
- Adopted the current `cookiecutter-scverse` template: `hatch`-managed test environments, `uv`-based CI and
  documentation builds, strict Pyrefly type checking and Ruff linting configured in `pyproject.toml`,
  local `pyproject-fmt`/`zizmor` hooks and prek's built-in file checks,
  and Dependabot updates.
- Reimplemented the existing generic reader as a package over the centralized
  lazy raster utilities. Its Python source argument is now `path`, image options
  are keyword-only, custom chunks can be passed through every generic image
  entry point, and the default requested chunk size is 1024 on the spatial axes
  only. Channel and depth axes retain their source chunks unless the caller
  explicitly describes them.
- Generic GeoJSON inputs now reject image-only options instead of warning and
  ignoring them. `generic_to_zarr()` no longer writes status messages to
  standard output; the CLI continues to report successful writes.
- **Breaking:** `xenium()` returns every element in a `global` coordinate system
  in micrometres instead of morphology pixels, uses `include_*` options,
  `cell_geometry`, `table_target` and `feature_types` (Gene Expression and
  Protein Expression by default), links the table through `label_id` or
  `cell_id`, and returns a single scale with storage chunks unless
  `scale_factors` or `chunks` are given. `xenium_aligned_image()` takes the
  pixel size explicitly and reads axes from OME metadata;
  `xenium_explorer_selection()` returns micrometre shapes indexed by selection.
  The `xenium` command mirrors the new options.
- **Breaking:** `iss()` and `steinbock()` take one `scale_factors` that applies
  equally to the image and its labels, replacing `image_scale_factors` and
  `labels_scale_factors` (CLI `--scale-factors`). Generated label pyramids
  still require labels of at most 32 bits.

### Removed

- `to_legacy_anndata(include_images=...)` and its image rendering/resampling
  path. Keep registered images and transformations in SpatialData.
- Support for Python 3.11.
- The generic image reader's `use_tiff_memmap` argument and its separate
  memmap/PIMS implementations. TIFF-family inputs now use one bounded,
  storage-aligned lazy path.
- The Xenium arguments `n_jobs`, `imread_kwargs`, `image_models_kwargs`,
  `labels_models_kwargs`, `gex_only`, `cells_as_circles` and the deprecated
  aliases, and the public cell-ID codec functions
  `cell_id_str_from_prefix_suffix_uint32()` and
  `prefix_suffix_uint32_from_cell_id_str()`. Cells tables, boundary files and
  count matrices already hold the decoded `cell_id`; to decode the
  `(prefix, suffix)` integer pairs of `cells.zarr.zip` yourself, apply the
  documented algorithm:

  ```python
  def cell_id(prefix: int, suffix: int) -> str:
      return "".join(chr(ord("a") + int(digit, 16)) for digit in f"{prefix:08x}") + f"-{suffix}"
  ```
