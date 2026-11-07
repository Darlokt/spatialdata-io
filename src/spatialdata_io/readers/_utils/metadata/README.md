# Structured metadata

This private package owns one strict JSON-object boundary for small metadata
used by reader layouts, and one strict text decoder for HDF5 attribute values:

```python
from spatialdata_io.readers._utils.metadata import decode_hdf5_text, parse_json_object, read_json_object
```

`read_json_object()` consumes an `ArtifactFile` already normalized by the
shared path package. It does not resolve, stat, or revalidate the path.
`parse_json_object()` applies the same contract to JSON text embedded in
another container, such as an HDF5 attribute, and names the storing artifact
in failures. The calling reader supplies a `ReaderErrorContext`.

## JSON contract

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

## HDF5 text attributes

`decode_hdf5_text(value, /, *, attribute, context, path)` turns one attribute
value, as h5py returns it, into nonempty UTF-8 text:

- `bytes` and `numpy.bytes_` (fixed-length strings) are decoded strictly;
- `str` and `numpy.str_` (variable-length strings) are first re-encoded with
  `surrogateescape`, because h5py decodes them with that handler: invalid
  UTF-8 such as `b"\xff"` arrives as `"\udcff"` and is rejected rather than
  accepted as text;
- a 0-d array is unwrapped to its element;
- any other value, invalid UTF-8 and empty text raise `ReaderFormatError` with
  the caller's context, the attribute name and the HDF5 path. NUL characters
  are not rejected.

The caller reads the attribute while it owns the open store; the decoder
never opens files. Multi-element arrays (for example 10x `library_ids`) and
the meaning of the text stay with the reader.

## Ownership boundaries

Readers retain artifact discovery, required keys, nested
schema validation, numeric ranges, units, and format-version interpretation.
The package does not open HDF5 or Zarr stores (it only decodes attribute values a reader already read), build GeoJSON geometry, read YAML,
remote streams or arbitrary JSON arrays. It contains no
vendor conditions, model construction, raster or tabular work, schema callback,
cache, or universal size limit.

The package root intentionally exports only `decode_hdf5_text()`,
`parse_json_object()` and `read_json_object()`. `_hdf5.py` and `_json.py`
contain the implementations; callers must not import their private helpers.
