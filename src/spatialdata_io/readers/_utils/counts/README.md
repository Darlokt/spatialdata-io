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

Both functions return the identical, freshly read `AnnData` after one shared
normalization and count-finalization pass. `X` is a finite, nonnegative,
integral, canonical float32 or float64 CSR sparse array. Canonical float32
Scanpy CSR storage can be wrapped without copying its buffers. Other
conversions allocate the required storage and never densify expression.
Sparse capacity beyond the terminal pointer is not an active contribution.

A repeated (barcode, feature) entry retained by Scanpy is malformed and raises
`ReaderFormatError`; HDF5 duplicates are never accepted as additive counts.
Normalization may reject invalid arithmetic before reporting the repeated
coordinate. The Matrix Market parser already sums repeated records before
returning, so those source repeats remain additive through Scanpy.

Integer count storage uses float32 when its maximum is at most 2**24,
otherwise float64 through 2**53; larger integer counts raise. Existing float32
or float64 storage is never widened and can be narrowed when the maximum is at
most 2**24. These guarantees concern the decoded values: Scanpy converts int32
HDF5 values to float32 before local validation, so rounded source integers
cannot be recovered. No expression records are reread and no source decoder
is replaced. Readers own feature selection, aggregation and Xenium protein
scaling after source loading.

The return value is eager, owned, and in memory. The public Scanpy readers own
expression and annotation parsing and return fully materialized tables. The
wrapper owns a short-lived MatrixMarket header stream and retains no open
resources, cache, temporary artifact, or process-global setting. Header inspection requires an
integer coordinate source; general and symmetric storage are accepted. Legacy
family selection follows Scanpy's uncompressed `genes.tsv` convention.
The upstream MTX parser can truncate malformed integer-header tokens and sum
duplicate records before returning. Result validation cannot recover that
information; strict rejection of every malformed MTX record is not guaranteed.

## Explicit options and side effects

The HDF5 boundary has no options and always passes `gex_only=False` and
`backup_url=None`; it retains every feature type and cannot download a missing
source. Multi-genome legacy HDF5 files are therefore rejected by Scanpy. The
Matrix Market boundary exposes `make_unique`, `prefix`, and `compressed`, uses
gene symbols as `var_names`, retains every feature type, and always passes
`cache=False`; it cannot create or consume a Scanpy cache. Readers own feature
selection.

An MTX prefix is the filename prefix the reader discovered in the validated
directory. Scanpy concatenates it with fixed companion basenames, so it cannot
contain a path separator.

Scanpy and AnnData continue to own storage-version interpretation, matrix
orientation, barcode/feature alignment, feature metadata, and name
uniqueness. Scanpy failures retain their original identity, type, cause, and traceback and
receive a reader/artifact/path note. Malformed count storage and values raise
`ReaderFormatError` with that context; unsupported normalization input types
retain `TypeError`.

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
