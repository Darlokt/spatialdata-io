# Sparse table boundary

This package is the sparse expression-representation boundary for
`spatialdata-io` readers. It turns reader-selected in-memory values,
coordinate records, or explicitly reader-owned `AnnData` expression storage
into canonical SciPy CSR sparse arrays.

## Responsibilities

The package owns:

- accepted in-memory dense and SciPy sparse representations;
- exact CSR sparse-array output;
- finite SciPy-supported real-valued expression data;
- two-dimensional shape and coordinate-bound validation;
- deterministic duplicate summation and sorted sparse indices;
- input non-mutation and explicit storage-sharing behavior;
- normalization of owned, in-memory `AnnData.X` and named expression layers;
- exact region, instance, and optional target linkage validation;
- one isolated call to the public `TableModel.parse()` boundary;
- rejection of implicit computation and densification.

A vendor reader owns:

- artifact, schema, and feature-column discovery;
- observation and feature ordering;
- identifier construction and semantic dtypes;
- filtering, joins, and the meaning of aggregation;
- choosing a value dtype wide enough for valid duplicate sums;
- deciding which layers semantically contain expression or counts;
- constructing linkage identifiers and selecting their semantic dtypes;
- selecting regions and independently known target indexes;
- SpatialData assembly.

## Package layout

- `_arrays.py` constructs canonical CSR sparse arrays without vendor or model
  dependencies.
- `_normalize.py` mutates explicitly reader-owned, in-memory `AnnData` objects
  without copying the complete table.
- `_linkage.py` validates reader-created linkage and optional exact target
  membership, normalizes expression storage, and calls the public
  `TableModel.parse()` API exactly once.

Only intentional reader-facing functions are re-exported from `table.__init__`.
Private modules should not become parallel reader APIs.

## Dense expression columns

Readers first select and validate the columns that semantically represent
expression, then make conversion to an in-memory NumPy array explicit:

```python
from spatialdata_io.readers._utils.table import as_csr_array

expression = counts.loc[observation_order, feature_order].to_numpy(copy=False)
x = as_csr_array(expression)
```

The helper does not accept a pandas object directly because doing so would hide
column interpretation and conversion behavior.

Readers should construct this CSR value before constructing `AnnData` whenever
they already own dense columns or coordinate triplets. This avoids holding an
avoidable second dense expression copy inside `AnnData` before normalization.

## Existing sparse data

SciPy sparse matrices and sparse arrays of any two-dimensional format are
normalized without densification:

```python
from scipy import sparse

from spatialdata_io.readers._utils.table import as_csr_array

legacy = sparse.csc_matrix([[1, 0], [0, 2]], dtype="int32")
x = as_csr_array(legacy)

assert isinstance(x, sparse.sparray)
assert not isinstance(x, sparse.spmatrix)
assert isinstance(x, sparse.csr_array)
assert sparse.issparse(x) and x.format == "csr"
```

## Owned AnnData normalization

Third-party readers such as `read_h5ad()` and Scanpy can return an in-memory
`AnnData` containing dense values or legacy sparse matrices. Once the reader
owns that table, normalize its primary expression and any explicitly documented
count layers in place:

```python
from spatialdata_io.readers._utils.table import normalize_owned_anndata

adata = normalize_owned_anndata(adata, expression_layers=("counts",))
```

`normalize_owned_anndata()` returns the identical object. It rejects views and
backed tables before accessing `X`, validates every requested layer name before
mutation, and then normalizes `X` and selected layers sequentially. Sequential
conversion bounds peak memory but is deliberately not transactional: if a later
layer is invalid, earlier values can already be normalized and the caller must
discard the failed owned table.

Unspecified layers, `raw.X`, annotations, aligned mappings, identifiers, and
ordering are unchanged. `raw.X` has no writable public boundary; a future
reader that intentionally exposes secondary expression should construct a
normalized named layer where practical rather than forcing an expensive rebuild
of `raw` here. `X=None` remains valid for metadata-only tables.

## Linked tables

`TableLinkage` carries the correlated parser arguments as immutable, ordered
state. Build final reader-owned region and instance columns before linkage,
then call `parse_linked_table()` once:

```python
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table

table = parse_linked_table(
    table,
    TableLinkage(("cell_shapes",), "region", "cell_id"),
    context=ReaderErrorContext(reader="example", artifact="cell table"),
    expression_layers=("counts",),
    expected_instance_ids={"cell_shapes": shapes.index},
)
```

For a multi-region table, descriptor order remains explicit and target
validation can cover only the independently known regions:

```python
table = parse_linked_table(
    table,
    TableLinkage(("sample_a", "sample_b"), "region", "cell_id"),
    context=ReaderErrorContext(reader="example", artifact="combined cell table"),
    expected_instance_ids={"sample_b": sample_b_shapes.index},
)
```

Every declared region must occur in the table, and `(region, instance)` pairs
must be unique. The same instance value may occur in different regions. Region
values are strings. Instance values are homogeneous strings or the fixed-width
integer dtypes accepted by the public SpatialData parser. Readers choose these
values and their semantic dtypes; the shared boundary never invents or rewrites
stored identifiers.

`expected_instance_ids` is optional and may cover a subset of regions. Supply
only independently known, already-materialized indexes such as a final shape or
point index or a small source mapping. Do not compute a labels raster merely to
enumerate identifiers. For every supplied region, target and table identifiers
must form exactly equal sets; order is irrelevant.

String and integer identifiers occupy distinct semantic families. Numeric
strings are not integers, booleans are not integers, and leading zeros remain
significant. Integer target indexes, including categorical indexes, may use a
different width or signedness when every value is losslessly representable by
the table's integer dtype. Temporary comparison arrays use that validated
dtype, avoiding pandas' lossy mixed signed/unsigned promotion. Borrowed target
indexes are never mutated.

Linkage validation precedes expression mutation. Missing or malformed linkage,
duplicate pairs, and target mismatches therefore leave the table unchanged.
Normalization and public parsing are deliberately non-transactional: a failure
at either boundary, or during final compatibility checks, can leave the owned
table partially mutated and the reader must discard it.

The public parser is called once, without `overwrite_metadata`. The adapter
does not import private SpatialData validators or write model metadata itself.
It verifies only stable public postconditions: object identity, retained
canonical CSR expression objects, categorical region storage, and exact parser
metadata.

## Errors

This representation boundary deliberately raises precise `TypeError` and
`ValueError` failures for unsupported storage, dtypes, shapes, coordinates, and
owned-`AnnData` state. Those failures describe direct low-level function inputs
and are not replaced by a generic reader exception.

Reader orchestration and later linkage or count-source utilities use
`ReaderErrorContext`, `ReaderFormatError`, and `add_reader_context` from
`spatialdata_io.readers._utils.errors` when an external artifact, schema, or
layout needs reader and version context. Raster storage retains the specialized
`RasterFormatError` from the same centralized module.

Malformed linkage values and target membership raise `ReaderFormatError` with
reader context. Invalid reader-authored descriptors, stale parser metadata, and
mapping keys outside the descriptor raise `ValueError`. Expression failures
retain the precise normalization exception and location note. Public parser
failures retain their original type, identity, traceback, and cause while
gaining reader context. A failed parser postcondition is reported as a local
compatibility `RuntimeError`.

## Sparse-array semantics

SciPy sparse arrays deliberately follow NumPy array semantics rather than the
legacy matrix semantics. Reader migrations must account for these differences
after construction:

- `*` is elementwise multiplication; use `@` for matrix multiplication;
- indexing one row, iteration, and many axis reductions can produce 1-D
  results, so use explicit two-axis indexing when a 2-D result is required;
- `.A`, `.H`, `getnnz()`, `getrow()`, and other matrix-only conveniences are
  unavailable; use the sparse-array alternatives documented by SciPy;
- `nnz` counts stored values, including explicit zeros, while
  `count_nonzero()` counts actual nonzero values;
- sparse-array constructors derive index dtypes from the supplied index-array
  dtypes rather than downcasting based on their values.

`csr_array_from_triplets()` therefore requires matching signed `int32` or
`int64` row and column arrays and preserves that dtype in the returned CSR
indices and index pointer. A shape too large for `int32` is rejected before
construction so SciPy cannot silently promote the structure. Readers should
choose `int32` when its range is sufficient and `int64` only for larger shapes
or an explicit interoperability requirement.

See SciPy's
[migration guide](https://docs.scipy.org/doc/scipy/reference/sparse.migration_to_sparray.html)
for the complete matrix-to-array behavior changes. Phase 2 owns construction;
each later reader cohort remains responsible for auditing its sparse indexing,
iteration, reductions, `*`, `@`, and power operations before adopting arrays.

## Coordinate triplets

Readers that already have observation, feature, and value vectors should avoid
constructing a dense intermediate:

```python
import numpy as np

from spatialdata_io.readers._utils.table import csr_array_from_triplets

rows = np.asarray([0, 0, 1], dtype=np.int32)
columns = np.asarray([1, 1, 0], dtype=np.int32)
values = np.asarray([2, 3, 4], dtype=np.int32)

x = csr_array_from_triplets(values, rows, columns, shape=(2, 2))
# The duplicate at (0, 1) is stored once with value 5.
```

Empty tables retain their explicit dimensions:

```python
empty = np.asarray([], dtype=np.float32)
indices = np.asarray([], dtype=np.int32)

no_observations = csr_array_from_triplets(empty, indices, indices, shape=(0, 8))
no_features = csr_array_from_triplets(empty, indices, indices, shape=(5, 0))
```

## Eager execution and ownership

SciPy sparse arrays are eager in-memory objects. Dask arrays, backed datasets,
masked arrays, GPU arrays, pandas objects, and generic `ArrayLike` values are
rejected before coercion. AnnData views are also rejected because assigning
their expression would implicitly materialize a full owned copy. Readers must
keep large projected CSV, Parquet, transcript, or event-table ingestion lazy at
its storage boundary, then pass only the intended materialized expression
representation here. Finite-value validation scans floating data in bounded
chunks, so it remains linear in stored values without allocating an additional
array proportional to the table size.

A canonical CSR array is returned unchanged. A canonical CSR matrix can be
wrapped without copying its buffers. Callers must explicitly copy the result
when independent mutable ownership is required. Noncanonical CSR inputs are
copied before sorting or duplicate summation, so this utility never changes an
input's data, indices, or structural flags.

The utility never calls `compute()`, `load()`, `to_memory()`, `toarray()`,
`todense()`, or an equivalent operation, and it never copies a complete
`AnnData`. It preserves SciPy-supported boolean and fixed-width integer dtypes
and `float32`, `float64`, or the platform `longdouble` dtype. SciPy does not
support `float16` sparse storage, so `float16` is rejected rather than silently
promoted. Each reader must select a value dtype capable of representing valid
aggregates.

Linkage validation constructs one in-memory pair index and performs a constant
number of linear scans over observation and supplied target identifiers. It
allocates no expression-sized validation array and performs no dataframe
`groupby`; the public SpatialData parser may perform its own model validation.
Canonical integer CSR arrays are reused. Canonical floating CSR arrays are also
reused, but their mutable stored values are rescanned for finiteness because no
trustworthy normalization marker is retained.

## Extending the package

Add a shared capability only when at least two concrete readers demonstrate the
same representation, safety, ownership, or linkage invariant. An extension
must:

1. keep vendor names, coordinate meaning, schema interpretation, identifier
   construction, and geometry policy outside this package;
2. expose one narrow, precisely typed API from `table.__init__`;
3. document mutation, copying, resource, eager/lazy, and failure behavior;
4. include deterministic unit tests for successful, empty, malformed,
   non-mutation, and no-densification paths;
5. add integration tests only when composing with `AnnData`, SpatialData models,
   or a source-family storage boundary;
6. pass strict Ruff, strict Pyrefly, and at least 90% targeted branch coverage.

Avoid generic reader callbacks, universal table loaders, untyped `ArrayLike`
inputs, `np.asarray()` before sparse-type detection, silent Dask computation,
COO outputs passed to `AnnData`, sparse-matrix outputs, and validation that
densifies expression data.
