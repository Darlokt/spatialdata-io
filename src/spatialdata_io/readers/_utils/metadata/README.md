# Bounded structured metadata

This private package owns one storage-level boundary for small JSON metadata
files used by reader layouts:

```python
from spatialdata_io.readers._utils.metadata import read_json_object
```

`read_json_object()` consumes an `ArtifactFile` already normalized by the
shared path package. It does not resolve, stat, or revalidate the path. The
calling reader supplies a positive, format-appropriate encoded byte limit and
a `ReaderErrorContext`.

## Contract

The function:

- performs one binary `read(max_bytes + 1)` and rejects oversized content
  before decoding or parsing;
- decodes strict UTF-8 and parses JSON exactly once;
- requires a top-level object and returns newly owned plain dictionaries;
- rejects duplicate decoded object names at every nesting level;
- rejects `NaN`, `Infinity`, `-Infinity`, and finite-syntax float overflow such
  as `1e400` during the parse;
- closes the source before content handling and retains no file-backed state;
- preserves open, read, and close `OSError` instances with contextual notes;
  and
- reports malformed external content with contextual `ReaderFormatError`.

The encoded payload, decoded text, and returned object require bounded
`O(max_bytes)` storage. Parsing is intentionally eager because layout, version,
scale, and transformation decisions need this small metadata immediately.
Sparse storage and delayed execution do not apply to this boundary.

## Ownership boundaries

Readers retain artifact discovery, byte-limit selection, required keys, nested
schema validation, numeric ranges, units, and format-version interpretation.
The package does not read embedded HDF5 or Zarr strings, GeoJSON geometry, YAML,
remote streams, arbitrary JSON arrays, or unbounded manifests. It contains no
vendor conditions, model construction, raster or tabular work, schema callback,
cache, or universal size limit.

The package root intentionally exports only `read_json_object()`. `_json.py`
contains the implementation; callers must not import its private helpers.
