# Typed 10x count sources

`spatialdata_io.readers._utils.counts` is the private source-family boundary
for 10x HDF5 and Matrix Market count containers. It deliberately wraps the
public Scanpy readers rather than reproducing their parsers.

## Contract

```python
from spatialdata_io.readers._utils.counts import read_10x_h5, read_10x_mtx
```

`read_10x_h5()` accepts an `ArtifactFile`; `read_10x_mtx()` accepts an
`ArtifactDirectory`. These nominal path types record prior canonicalization,
containment policy, existence, and artifact-kind validation. The count
boundary borrows them and performs no second `stat()`, resolution, or kind
check.

Both functions return the identical, freshly read `AnnData` after
`normalize_owned_anndata()` has converted `X` to a finite canonical SciPy CSR
sparse array. A canonical Scanpy CSR matrix can be wrapped without copying its
`data`, `indices`, or `indptr` buffers. Conversion from another sparse format
allocates only the required CSR storage and never densifies expression data.

The return value is eager, owned, and in memory. The public Scanpy readers own
the source lifetime and return fully materialized tables. The wrapper opens or
retains no file, HDF5 object, cache, temporary artifact, or process-global
setting.

## Explicit options and side effects

The HDF5 boundary exposes only `genome` and `gex_only` and always passes
`backup_url=None`; it cannot download a missing source. The Matrix Market
boundary exposes `var_names`, `make_unique`, `gex_only`, `prefix`, and
`compressed` and always passes `cache=False`; it cannot create or consume a
Scanpy cache.

An MTX prefix is portable filename text, not a path. ASCII control characters,
POSIX and Windows separators, Windows-reserved characters, drives, alternate
data streams, and generated DOS device names are rejected before Scanpy is
called. The empty prefix and literal dotted text, including `.` and `..`, are
safe: Scanpy concatenates the prefix with fixed companion basenames.

Scanpy and AnnData continue to own storage-version interpretation, matrix
orientation, barcode/feature alignment, feature metadata, filtering, and name
uniqueness. Their failures and normalization failures retain their original
identity, type, cause, and traceback and receive a reader/artifact/path note.

## Deliberate exclusions

This package does not infer library IDs, read HDF5 root attributes, install
`uns["spatial"]`, scale Xenium proteins, aggregate Visium HD observations,
construct identifiers, link tables, or attach provenance. Readers perform
those semantic operations after source loading and use final linkage as the
authoritative post-mutation normalization boundary.

Delayed or backed tables are also excluded. Scanpy's supported public 10x
readers are eager, and the shared table/linkage contract requires an owned
mutable in-memory `AnnData`. A future out-of-core table path would need its own
measured representation, model compatibility, and resource-lifetime contract;
it must not be hidden behind these functions.

This package remains private and is not exported from the lazy
`spatialdata_io` root. Importing its public table-normalization dependency also
loads the cohesive table package, including its SpatialData linkage adapter.
That transitive import is accepted: every intended Visium, Visium HD, and
Xenium workflow requires SpatialData to assemble its result, so delaying the
same dependency would not reduce successful-reader peak memory. `_10x.py`
nevertheless has no direct linkage, SpatialData-model, vendor, geometry, or
raster dependency.
