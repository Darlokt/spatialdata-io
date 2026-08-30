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

### Changed

- Adopted the current `cookiecutter-scverse` template: `hatch`-managed test environments, `uv`-based CI and
  documentation builds, `mypy` type checking of `src` and `tests`, `biome`/`pyproject-fmt`/`zizmor` pre-commit hooks,
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

### Removed

- Support for Python 3.11.
- The generic image reader's `use_tiff_memmap` argument and its separate
  memmap/PIMS implementations. TIFF-family inputs now use one bounded,
  storage-aligned lazy path.
