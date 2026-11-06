# Table representation boundary

This package is the expression and annotation representation boundary for
`spatialdata-io` readers. It turns reader-selected in-memory values,
coordinate records, or explicitly reader-owned `AnnData` expression storage
into canonical SciPy CSR sparse arrays.
It also prepares explicitly owned annotation frames for AnnData storage and
validates final SpatialData table linkage.

## Responsibilities

The package owns:

- accepted in-memory dense and SciPy sparse representations;
- exact CSR sparse-array output;
- finite SciPy-supported real-valued expression data;
- two-dimensional shape and coordinate-bound validation;
- deterministic duplicate summation and sorted sparse indices;
- input non-mutation and explicit storage-sharing behavior;
- normalization of owned, in-memory `AnnData.X` and named expression layers;
- supported Arrow annotation storage conversion in explicitly selected owned frames;
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
  to normalize expression without copying the complete table.
- `_compat.py` prepares explicitly selected owned annotation frames for AnnData
  storage without inspecting expression or recursively walking annotations.
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

Pandas is one possible input container, not a required expression-processing
stage. A reader may instead select validated Arrow columns or native numerical
arrays and construct its NumPy block directly. Keep annotation alignment in
pandas when needed; do not wrap expression columns in a DataFrame or convert
native numerical buffers to Arrow solely for uniform intermediate storage.

For dense-on-disk CSV expression, the reader owns the declared parsing contract
and bounded numerical batches, followed by CSR assembly. Budget numerical
blocks using feature count and dtype width, and account separately for parser
and string buffers. Final CSR and observation annotations remain proportional
to input size. Neither an Arrow backend nor `copy=False` proves that a conversion
avoids allocation; validate that property for the actual path before claiming it.

Readers should construct this CSR value before constructing `AnnData` whenever
they already own dense columns or sparse coordinates. This avoids holding an
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

CSR and CSC arrays and matrices have their compressed buffers validated before
conversion or canonicality shortcuts. `validate_compressed_sparse()` exposes
the same read-only structural check when a reader must inspect compressed
spans before an operation that trusts them, such as row reordering. Final
linkage repeats the check, so a reader needs it only for that purpose. It checks one-dimensional NumPy buffers,
integer indices/pointers safely representable by int64, equal data/index
capacity, pointer length and initial zero, nondecreasing pointers, terminal
extent, and active minor-axis bounds.
Unsorted indices and duplicates are valid; reader-specific count and aggregate
policies remain reader-owned.

The validator preserves borrowed buffers and cached flags, does not call
SciPy's mutating `check_format()`, and does not copy expression or construct a
second matrix. Equal unused data/index capacity after the terminal pointer is
valid; inactive indices are not inspected. Pointer comparisons and active-index
scans use bounded workspace. Structural validation never trusts sorting or
canonicality flags, which do not establish that external indices are in bounds.
Signed and unsigned integer widths supported by SciPy, including nonnative byte
order, remain valid. Unsigned 64-bit indices/pointers are rejected before
canonicality inspection: SciPy's compressed kernels cannot safely dispatch that
storage. Expression value dtypes are unaffected.

Owned-AnnData normalization always combines compressed integer duplicates
losslessly, with floating promotion when the source dtype is insufficient.
Generic `as_csr_array()` retains its dtype-preserving arithmetic contract;
see [imported sparse duplicates](#imported-sparse-duplicates).

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

## Owned annotation frames

Reader schemas own column semantics, nullability, ordering, and validation.
After final annotation mutations and semantic validation, call
`prepare_anndata_dataframe()` once on each final owned AnnData-bound frame,
before constructing or linking the table:

```python
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.table import prepare_anndata_dataframe

obs = prepare_anndata_dataframe(
    obs,
    context=ReaderErrorContext(reader="example", artifact="observation annotations"),
)
```

This temporary compatibility boundary replaces general Arrow extension storage
unsupported by AnnData writers or SpatialData integer linkage:

| Arrow input | Final storage |
| --- | --- |
| Signed/unsigned integers without nulls | Same-width, same-signedness NumPy integers |
| Signed/unsigned integers with nulls | Matching pandas nullable integer extension |
| Boolean without/with nulls | NumPy bool / pandas nullable boolean |
| Float32/64 without nulls | Matching NumPy floating dtype, preserving valid NaNs |
| Float32/64 with nulls | Rejected unless the reader schema authorizes null-to-NaN conversion |
| String / large string | pandas `StringDtype(storage="pyarrow")`, backed by `ArrowStringArray` |
| Decimal, nested, binary, temporal, other Arrow storage | Contextual `ReaderFormatError` |

The function mutates and returns the identical eager frame. Compatible columns,
including existing pandas string extensions and categoricals, retain their
arrays. Column labels/order, lexical identifiers, integer extrema including
uint64, categorical order, index names/order, and MultiIndex codes are
preserved. Arrow indexes and MultiIndex levels follow the column conversion
rules. AnnData itself restricts supported observation/feature index types;
this helper does not stringify numeric indexes or permit MultiIndex in AnnData.
It accepts only an explicitly owned frame, never copies complete AnnData,
touches expression, reads files, changes global settings, or traverses nested
annotations. Readers explicitly select obs, var, and any schema-declared
DataFrame in uns. Preserve imported non-Arrow annotations in their existing
representations. Conversion is sequential: discard the owned destination after
failure rather than attempting rollback.

Arrow distinguishes a null from a valid NaN. By default any floating null is
rejected, including an all-null column. A reader may pass
`allow_float_nulls=True` only when its semantic schema equates both states with
missing floating values. That explicit choice converts nulls to NaN while
retaining existing valid NaNs, width, and finite values. Readers requiring the
distinction must keep the default rejection until a supported lossless
representation is established.

Supported AnnData (>=0.13) and pandas (>=3.0) versions persist nullable pandas
string extensions under their default settings. The helper does not override
caller settings. Round-trip tests use public H5AD/Zarr writers and verify
exact integer widths/values, missingness, strings, indexes, and linkage, with
two concrete reader-shaped annotation fixtures. AnnData's independent
automatic string-to-category option remains a writer choice.

Keep this conversion isolated from `normalize_owned_anndata()`'s expression-only
contract and `parse_linked_table()`'s linkage validation. Remove it and reader
calls when supported upstream versions pass direct Arrow annotation persistence,
exact integer linkage, and reader composition checks. An upstream support
announcement alone does not establish those contracts.

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
at either boundary can leave the owned table partially mutated and the reader
must discard it.

The public parser is called once, without `overwrite_metadata`. The adapter
does not import private SpatialData validators or write model metadata itself.
Compatibility with the parser's object identity, categorical region storage,
expression preservation, and linkage metadata is covered through real public
SpatialData composition tests rather than repeated after every reader call.

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
the reader context and the optional source `path` passed to
`parse_linked_table()`. Invalid reader-authored descriptors and mapping keys
outside the descriptor raise `ValueError`. Stale imported linkage metadata,
expression normalization failures and public parser failures retain their
original type, identity, traceback, and cause while gaining a note naming the
reader context and path, so readers need no wrapper around the call.

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

See SciPy's
[migration guide](https://docs.scipy.org/doc/scipy/reference/sparse.migration_to_sparray.html)
for the complete matrix-to-array behavior changes. This package owns construction;
each reader remains responsible for auditing its sparse indexing,
iteration, reductions, `*`, `@`, and power operations before adopting arrays.

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
Compressed structure is similarly revalidated on each public conversion call,
including calls from owned-AnnData normalization and final linkage.

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

## Count accumulation and completed families

`count_accumulation_dtype(maximum_sum)` selects arithmetic from a nonnegative
Python-integer bound on every partial sum: float32 up to and including 2²⁴,
otherwise float64 up to and including 2⁵³. Larger bounds return `None`, meaning
exact floating accumulation is unproven. Readers validate source contributions,
derive bounds without fixed-width overflow, and select their own aggregation and
fallback behavior. Stereo-seq derives its bound from native gene spans and can
fall back to exact integer reduction; Visium HD derives its bound from nucleus
membership and raises a contextual precision error when the bound is inconclusive.
The helper rejects booleans, non-Python integers and negative bounds; it performs
no array scans, I/O, aggregation or mutation.

`finalize_count_arrays((X, exon), context=...)` accepts nonempty families of
owned canonical CSR arrays after all count arithmetic. It validates finite,
nonnegative integral values in bounded stored-buffer batches and chooses one
common float32 dtype when every actual value is exact, otherwise float64.
Batches whose largest value is at most 2²⁴ need no element-wise cast check,
which keeps ordinary counts to one validation pass.
Counts that neither dtype preserves raise `ReaderFormatError`. This is storage
conversion, not scaling. Readers pass `path=` to identify the source artifact
in numerical errors. It preserves explicit zeros, order, shapes and sparse
index widths. Suitable floating arrays are returned unchanged; converted arrays
own new values and may share owned structural buffers. No input is mutated.
The helper does not parse source text, sum duplicates or recover upstream loss.

## Imported sparse duplicates

`normalize_owned_anndata()` and `parse_linked_table()` always use lossless
integer duplicate normalization for CSR/CSC expression. An optional reader-owned
`duplicate_sums=[]` collector records summaries without choosing arithmetic or
accepted representations. Owned expression supports dense NumPy and CSR/CSC
storage; other sparse formats use the separate generic conversion boundary.
Repeated matrix coordinates are
summed, not discarded. Integer storage stays unchanged when completed sums fit;
otherwise it promotes directly to float32 when every completed value is exact,
or float64 if needed. No wider integer array is staged before floating output.
Unaffected coordinates must also fit the selected floating dtype. Exact
Python-integer totals prevent source rounding before promotion. No supported
lossless floating result raises a
contextual normalization error. Floating and boolean expression retain their
existing numerical semantics. This does not impose raw-count semantics.

Storage is validated before reading cached canonicality. A fresh compressed
wrapper checks actual indices without mutating source metadata or buffers.
Compressed-span bounds use bounded pointer batches and retain native integer
storage whenever sufficient. Proven floating bounds enable SciPy reduction;
exceptional inputs use dictionaries bounded by one compressed row/column.
There is no densification or object-valued expression result. Unsorted unique
indices are merely sorted and do not produce a duplicate report. Combined
entries, including explicit-zero sums, append typed `SparseDuplicateSummary`
values identifying the slot, number combined and source/result dtypes.

`normalize_sparse_duplicates()` exposes the same numerical reduction without
AnnData slot selection or warning collection. Readers can reuse it for expected
format aggregation when their ordinary arithmetic bound cannot prove safety;
it does not itself emit warnings or infer nonnegative count semantics.

The reader owns reporting: emit one `UserWarning` from accumulated summaries
only after successful result assembly, with affected artifacts/tables/slots.
Expected format aggregation is not an imported-data warning. Omitting the
collector does not change normalization. Unnamed layers and raw remain
outside the reader's explicitly declared normalization slots.
