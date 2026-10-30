# Structured metadata

This private package owns one storage-level boundary for small JSON metadata
files used by reader layouts:

```python
from spatialdata_io.readers._utils.metadata import read_json_object
```

`read_json_object()` consumes an `ArtifactFile` already normalized by the
shared path package. It does not resolve, stat, or revalidate the path. The
calling reader supplies a `ReaderErrorContext`.

## Contract

The function:

- consumes one owned strict UTF-8 text stream through the standard JSON decoder;
- requires a top-level object and returns newly owned plain dictionaries;
- rejects duplicate decoded object names at every nesting level;
- rejects `NaN`, `Infinity`, `-Infinity`, and finite-syntax float overflow such
  as `1e400` during the parse;
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
The package does not read embedded HDF5 or Zarr strings, GeoJSON geometry, YAML,
remote streams or arbitrary JSON arrays. It contains no
vendor conditions, model construction, raster or tabular work, schema callback,
cache, or universal size limit.

The package root intentionally exports only `read_json_object()`. `_json.py`
contains the implementation; callers must not import its private helpers.
