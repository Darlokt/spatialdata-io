# Reader provenance

`spatialdata_io.readers._utils.provenance` is the private boundary for
recording stable `spatialdata-io` reader provenance on a newly assembled
reader result. It operates on an attribute mapping rather than importing or
returning `SpatialData`.

## Contract

```python
from spatialdata_io.readers._utils.provenance import set_reader_provenance

set_reader_provenance(
    sdata.attrs,
    reader="xenium",
    format_version="3.0.1",
)
```

The function borrows a `MutableMapping[str, object]`, mutates it in place, and
returns `None`. It owns exactly three keys:

- `spatialdata_io_software_version`, set from the canonical package version;
- `spatialdata_io_reader`, set from the reader-authored normalized name; and
- `spatialdata_io_format_version`, set only for an already-normalized and
  reliably detected vendor format or software version.

Passing no format version removes a stale owned format-version entry. Existing
values of the other owned keys are replaced, while unrelated attributes and
mapping identity are preserved. Native mapping failures propagate unchanged;
the function does not copy the mapping or provide transactional rollback.

## Ownership and execution

Reader or layout code owns format discovery, parsing, normalization, and the
decision that a source version is reliable. The provenance boundary accepts
those internally authored strings without redundant runtime validation. It
does not read metadata or inspect paths, artifacts, credentials, or caller
options.

The operation is constant-time and independent of dataset size: two assignments
plus one optional assignment or deletion. It performs no pixel or table access,
creates no Dask task, touches no sparse representation, and owns no resource.

## Package layout

- `_core.py` owns the mapping mutation and its closed three-field schema.
- `__init__.py` re-exports only `set_reader_provenance()`.

The package remains private and is not exported from the lazy
`spatialdata_io` root or the mixed `readers._utils` package root.

## Deliberate exclusions

The schema does not include absolute paths, artifact names, checksums,
credentials, secrets, reader options, or a speculative schema version. The
package has no SpatialData, AnnData, pandas, NumPy, Scanpy, Dask, HDF5, Zarr,
raster, path, contextual-error, or vendor-reader dependency. It does not
assemble a `SpatialData` object or provide a reader registry, version parser,
provenance dataclass, nested schema, or compatibility wrapper.
