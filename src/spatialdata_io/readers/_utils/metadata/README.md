# Structured metadata

This private package owns one strict JSON-object boundary for small metadata
used by reader layouts:

```python
from spatialdata_io.readers._utils.metadata import parse_json_object, read_json_object
```

`read_json_object()` consumes an `ArtifactFile` already normalized by the
shared path package. It does not resolve, stat, or revalidate the path.
`parse_json_object()` applies the same contract to JSON text embedded in
another container, such as an HDF5 attribute, and names the storing artifact
in failures. The calling reader supplies a `ReaderErrorContext`.

## Contract

The function:

- consumes one owned strict UTF-8 text stream through the standard JSON decoder;
- requires a top-level object and returns newly owned plain dictionaries;
- rejects duplicate decoded object names at every nesting level;
- rejects `NaN`, `Infinity`, `-Infinity`, finite-syntax float overflow such
  as `1e400`, and integers beyond the float range during the parse, so readers
  can convert every accepted number to `float`;
- closes the source on success and failure and retains no file-backed state;
- preserves open, read, and close `OSError` instances with contextual notes;
  and
- reports malformed external content with contextual `ReaderFormatError`.

Parsing is intentionally eager because layout, version, scale, and
transformation decisions need the metadata immediately. There is no arbitrary
file-size acceptance limit. The standard decoder materializes the text and
decoded object; it is not an incremental JSON parser.
Sparse storage and delayed execution do not apply to this boundary.

## Ownership boundaries

Readers retain artifact discovery, required keys, nested
schema validation, numeric ranges, units, and format-version interpretation.
The package does not open HDF5 or Zarr stores, build GeoJSON geometry, read YAML,
remote streams or arbitrary JSON arrays. It contains no
vendor conditions, model construction, raster or tabular work, schema callback,
cache, or universal size limit.

The package root intentionally exports only `parse_json_object()` and
`read_json_object()`. `_json.py`
contains the implementation; callers must not import its private helpers.
