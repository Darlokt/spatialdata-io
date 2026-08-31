# Sparse table boundary

This package is the sparse expression-representation boundary for
`spatialdata-io` readers. It turns reader-selected in-memory values or
coordinate records into canonical SciPy CSR sparse arrays.

## Responsibilities

The package owns:

- accepted in-memory dense and SciPy sparse representations;
- exact CSR sparse-array output;
- finite SciPy-supported real-valued expression data;
- two-dimensional shape and coordinate-bound validation;
- deterministic duplicate summation and sorted sparse indices;
- input non-mutation and explicit storage-sharing behavior;
- rejection of implicit computation and densification.

A vendor reader owns:

- artifact, schema, and feature-column discovery;
- observation and feature ordering;
- identifier construction and semantic dtypes;
- filtering, joins, and the meaning of aggregation;
- choosing a value dtype wide enough for valid duplicate sums;
- expression-layer selection;
- table linkage and SpatialData model construction.

## Package layout

- `_arrays.py` constructs canonical CSR sparse arrays without vendor or model
  dependencies.
- A later `_normalize.py` may normalize explicitly reader-owned `AnnData`
  objects after that contract is independently implemented and tested.
- A later `_linkage.py` may validate table-to-element linkage and call
  `TableModel.parse()` after that contract is independently implemented and
  tested.

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
masked arrays, pandas objects, and generic `ArrayLike` values are rejected
before coercion. Readers must keep large projected CSV, Parquet, transcript, or
event-table ingestion lazy at its storage boundary, then pass only the intended
materialized expression representation here. Finite-value validation scans
floating data in bounded chunks, so it remains linear in stored values without
allocating an additional array proportional to the table size.

A canonical CSR array is returned unchanged. A canonical CSR matrix can be
wrapped without copying its buffers. Callers must explicitly copy the result
when independent mutable ownership is required. Noncanonical CSR inputs are
copied before sorting or duplicate summation, so this utility never changes an
input's data, indices, or structural flags.

The utility never calls `toarray()`, `todense()`, or an equivalent operation.
It preserves SciPy-supported boolean and fixed-width integer dtypes and
`float32`, `float64`, or the platform `longdouble` dtype. SciPy does not support
`float16` sparse storage, so `float16` is rejected rather than silently
promoted. Each reader must select a value dtype capable of representing valid
aggregates.

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
