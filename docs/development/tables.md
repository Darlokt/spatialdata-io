# Implement tables and transcript points

## Tabular sources and execution

Declare Arrow types and field roles in the reader's `_schemas.py`. Pair a
version-specific artifact with its types in layout discovery. Derive dynamic
features from the validated header and evidence that identifies expression
columns. Validate missing, duplicate and additional columns, support arbitrary
order for named fields where the format allows it, and retain any documented
positional roles.

| Source | Storage and parsing |
| --- | --- |
| Eager CSV annotations | Read headers through `csv.csv_header()` with duplicate-name rejection. Use `pyarrow.csv.read_csv()` with explicit projection, types and null options. Wrap the read in `csv.csv_errors()`. |
| Wide CSV expression | Use `pyarrow.csv.open_csv()` with an explicit source block size and the same conversion options. Wrap iteration in `csv.csv_errors()`. Convert each typed batch to sparse storage before reading the next batch. |
| Lazy point CSV | Use `csv.read_csv_blocks()` with the full validated header and projected schema for uncompressed sources. The physical-record decoder `csv.parse_csv_records()` also accepts independent decompressed partitions. `csv.open_csv_stream()` owns sequential plain or gzip streams. |
| Parquet | Inspect the footer and physical types with Arrow, project supported columns and use checked casts or projected batches. |
| Native numerical inputs | Keep HDF5 numeric data, FCS buffers and expression batches in Arrow or native numerical arrays until numerical or sparse construction. |

For eager CSV reads, set `column_types`, `include_columns`, `null_values`
and `strings_can_be_null` explicitly. With `null_values=[]` and
`strings_can_be_null=False`, text retains its literal spelling, including
empty strings and strings such as `NA`. Validate required values separately
from Arrow's schema metadata. Enable `newlines_in_values` for eager formats
that permit quoted multiline records. Shared lazy CSV partitions use physical
lines, so multiline records fail on computation. An oversized record may
exceed the target size of a lazy CSV block. For eager numerical batches,
record the source block size, required record bound and declared value width
when estimating peak allocation.

Arrow's declared CSV types define numeric conversion. Integer fields trim
spaces and tabs and accept hexadecimal. Decimal/exponent spellings for
integers, a leading plus sign and digit separators fail. Values outside the
declared decimal type or unsigned hexadecimal range fail. Signed hexadecimal
may wrap, for example int64 `0xffffffffffffffff` becomes `-1`. Where that
could be mistaken for a meaningful sentinel, validate the source spelling as
text and cast with overflow checks. Floating fields round directly to their
declared width, including underflow to zero. Parsed `nan` and `inf` require
an explicit finiteness check where the format requires finite values.
Validate other semantics such as positive identifiers, uniqueness and target
membership at the boundary that owns them.

Build pandas annotations from Arrow with `types_mapper=pd.ArrowDtype`. Use
Arrow-backed indexes and columns for alignment until the destination requires
a concrete representation. For `PointsModel`, convert coordinates to NumPy
numbers and supply known categories for a categorical feature key. The
`points.categorize_features()` helper uses an existing vocabulary's order
and returns `None` for unknown or missing values. The reader decides what
those values mean. For `AnnData.obsm` and expression, use supported NumPy and
SciPy storage.

## Expression, annotation storage and linkage

| Boundary | Contract |
| --- | --- |
| `counts` | Use the shared 10x HDF5 and Matrix Market decoders for supported sources. HDF5 rejects duplicate matrix coordinates. Matrix Market duplicate records are summed exactly. Readers own feature selection, library metadata, additional aggregation and identifier semantics. |
| `table.as_csr_array()` | Establish canonical finite CSR sparse-array storage. Use sparse arrays during numerical work. Keep row and feature order and avoid densifying expression. |
| `table.count_accumulation_dtype()` and `table.finalize_count_arrays()` | Prove exact nonnegative integer accumulation using a bound on the sum, and choose the final storage after arithmetic and filtering. Pass related count arrays, such as genes and blanks, together. Integer and mixed storage use float32 through 2²⁴ and float64 through 2⁵³, and raise above that range. A family already stored in one floating dtype is validated at source precision and may be narrowed when the values fit. |
| `table.normalize_owned_anndata()` | Normalize `X` and the specified expression layers of an owned in-memory table. Dense numeric arrays and CSR/CSC storage are accepted. Integer duplicates are additive and must reduce and promote exactly. An optional summary collector allows the reader to report unexpected duplicate normalization after successful assembly. Unrelated layers, aligned mappings and imported `raw` keep their source semantics. |
| `table.prepare_anndata_dataframe()` | After the final annotation mutations, adapt the owned Arrow-backed `obs` or `var` frame and index once for AnnData. Preserve integer width and signedness, text and missingness. Permit conversion of Arrow floating nulls to NaN only where the schema explicitly equates them. |
| `table.parse_linked_table()` | After filtering, order, identifiers and targets are final, link each table once using `TableLinkage`. The declared regions must occur and region/instance pairs must be unique and non-null. Supply independently available target indexes to validate exact set membership per region. For lazy labels, use available metadata such as the label dtype and keep pixel reads lazy. |

Choose identifier widths from the format. Validate before casting. A table
with uint64 instance IDs cannot be concatenated directly with an int64
instance column, because NumPy promotes the pair to float64. Source domains
that fit in uint16 or uint32 can use those widths and combine with int64
without promotion to floating point. [Table linkage and storage](../user/tables.md)

## Write an AnnData position declaration

Use `spatialdata_io._positions.PositionsMetadata` and `POSITIONS_KEY` when
constructing an owned reader table. Align coordinate rows to the **final**
`obs` order. Canonical `obsm["spatial"]` requires finite float64 XY micrometres,
the source definition of the position, and an established frame for every row.
For example, Xenium records the vendor cell centroid in a single frame:

```python
from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata

positions: PositionsMetadata = {
    "schema_version": 1,
    "coordinates": {
        "spatial": {
            "axes": ["x", "y"],
            "units": "micrometer",
            "position": "vendor cell centroid",
            "frame": "global",
            "calibration": {"global": "Xenium cells x_centroid/y_centroid in micrometres"},
        }
    },
}
table.uns[POSITIONS_KEY] = positions
```

For row-specific frames, supply `frame_key` in place of `frame`. It names a
categorical string observation column. Key physical calibration evidence by
frame. Keep pixel or grid positions under their respective native keys, or in
documented source columns. Preserve an unverified imported `spatial` array
exactly as `spatial_source` without inventing a declaration for it. Check for
occupied destinations before writing. If any contributing sample lacks
calibration, retain all rows and their native data and omit canonical `spatial`.

Choose the conversion that matches the source definition. For zero-based
regionprops centroid indices, `centroids.scale_regionprops_centroids()` shifts
to pixel centres and applies validated XY spacing on an owned float64 array.
Vendor-reported cell centres can have a different origin. Test native and
physical values, XY order, frame membership and preservation after a Zarr round
trip. Table linkage is recorded separately in `uns["spatialdata_attrs"]`.
[AnnData fields in reader output](../user/tables.md#where-to-find-annotations-and-positions)
