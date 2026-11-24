# Reader implementation contracts

Dataset readers in `src/spatialdata_io/readers/` use the private
`spatialdata_io.readers._utils` package for storage and validation common
to multiple formats. Reader code owns artifact names, layout and version
interpretation, axes and units, feature and observation order, geometry
and registration, semantic identifiers, and the returned element names.

## Choose a shared utility

- For HDF5 or Matrix Market counts using a supported 10x encoding, start in
  [`counts`](tables.md#expression-annotation-storage-and-linkage). For an imported
  H5AD, load an owned table and use `table.normalize_owned_anndata()`.
  For a format with its own numerical layout, decode its sparse records in
  the reader and use `table` for canonical storage and linkage.
- For tabular text, use [`csv`](tables.md#tabular-sources-and-execution) for common
  stream, header and error handling. The reader declares the fields and
  chooses a bounded eager read or a lazy points read based on the product.
- For TIFF, PNG or JPEG, use [`image`](rasters.md) for codecs,
  metadata inspection and lazy I/O. The reader decides where the image
  belongs and what its axes, channels and calibration mean.

The shared boundary gives formats the same validation and resource behavior.
For example, `table` can check that table IDs exactly match a supplied shape
index. The reader knows what the IDs denote, which shapes they refer to and
whether a geometry drop removes table rows.
[Visium walkthrough](visium-walkthrough.md)

Use the intentional exports of each utility package. For exact input,
exception and ownership details, consult the function docstrings and the
README files in `src/spatialdata_io/readers/_utils/`. The corresponding tests
live under `tests/unit/readers/_utils/` and
`tests/integration/readers/_utils/`.

Keep a new rule in the reader when it represents a vendor key, filename
precedence, scientific convention or product selection. Extend a utility
when at least two concrete readers need the same behavior, with the same
inputs, error and ownership contract. Exercise the common behavior in the
utility's tests and verify those tests before changing the readers to use it.
This keeps a format-specific workaround from becoming a dependency for other
formats.

When two vendor entry points read the same on-disk format, give its metadata
parser and image builder one owner. For example, CODEX Instrument Manager and
PhenoCycler both use the Akoya QPI format. They can call the same private QPI
implementation with validated paths and normalized raster options. TIFF IFD
access and resource lifetimes belong in `image`. The meanings of
`FullResolution`, `Biomarker` and vendor intensity units belong with QPI.

## Artifact paths, errors and provenance

| Boundary | Contract |
| --- | --- |
| `paths` | Use `normalize_dataset_root()` once and carry validated `DatasetRoot`, `ArtifactFile` and `ArtifactDirectory` values. Use caller-path helpers for explicit overrides, metadata-path helpers for discovered artifacts and dataset-provided paths, and sorted listing or unique-match helpers for discovery. Carry a logical filename separately when format, sample or channel identity depends on it. |
| `metadata` | Use `read_json_object()` for small JSON metadata files and `parse_json_object()` for bounded embedded text. Both require a top-level object, unique keys and finite numbers. Use `decode_hdf5_text()` for strict UTF-8 HDF5 metadata. Validate vendor keys and values in the reader. Use a safe loader for YAML and own the open source with a context manager. |
| `errors` | At external boundaries, bind the reader, artifact and known version with `ReaderErrorContext`. `ReaderFormatError` reports violated layout, format or schema requirements with the observed and expected state. Use an exception note via `add_reader_context()` when preserving the original exception type, or explicit chaining when translating a parser error. |
| `provenance` | `set_reader_provenance()` writes the reader, package version and reliably detected optional format version to the result's `attrs`. Keep local paths, credentials and eager artifact hashes out of provenance. |

During discovery, interpret the selected filename and carry its meaning alongside
the validated path. For example, a CODEX link named `reg001_compensated.csv`
declares both the CSV encoding and the compensated signal product. Table parsing
and output metadata consume those layout fields. Test with a symlink whose
target has a different suffix or product name.

Keep the path-kind exceptions and their `errno` and `filename` where
available. Report invalid public option types or values with the appropriate
`TypeError` or `ValueError`. Use regular exceptions for external validation,
and carry validated facts forward as typed private values. Keep files and
stores under explicit resource owners, including on error. [Dataset paths
and symlinks](../user/paths.md)

## Element and annotation names

`names.validate_element_name()` checks explicit caller names. For names
derived from source files, `names.sanitize_element_name()` repairs invalid
names and warns with the original and resulting values. The same module
checks case-insensitive collisions and can sanitize imported annotation
columns or top-level `uns` keys. Supply the reserved names, choose which
imported metadata can be renamed and check final product names before
constructing elements.
