# Table inputs and outputs

Visium, Curio, CODEX, ISS, Steinbock, and DBiT validate their tabular inputs
before constructing linked SpatialData tables. The accepted columns and
identifier meanings remain specific to each format.

## Input fields

Named fields are selected by name. Required fields must be present; optional
fields are validated when present. Reordering named columns is accepted where
the format supports named fields. Additional columns are either preserved as
annotations or explicitly ignored. Ambiguous duplicate column names fail.
Headerless files retain their positional field counts and meanings.

Visium headered tissue positions accept the six documented fields in any
order and ignore additional fields. Headerless positions require exactly six
fields. Barcode order is aligned to the expression table, including spots
outside tissue. Grid coordinates must be nonnegative integers fitting int64.
Pixel coordinates must be finite, nonnegative float64 values; conversions
that lose large integer precision or underflow to zero fail.

Curio metric exports have a positional exception: their first two columns
contain the metric name and value. A subsequent named `Category` column selects
the metric group. These first two columns are not freely reorderable. CODEX
expression features are recognized by their cycle/channel names; additional
metadata columns are not automatically treated as expression. DBiT barcode
maps retain their two-column positional contract.

## Annotation storage

Reader-created tabular intermediates use explicit Arrow-backed columns.
Imported H5AD observation and feature annotations retain their source types
unless the reader documents a required normalization. Additional annotations
are preserved according to the reader's format contract. Leading zeros in
string identifiers, integer signedness and width, and observation and feature
order remain significant.

Final AnnData annotations use storage compatible with the supported AnnData
and SpatialData versions:

| Annotation | Final storage |
| --- | --- |
| Required integer fields | NumPy integers with the declared width and signedness |
| Nullable integer annotations | Matching pandas nullable integer types |
| Boolean annotations | NumPy bool or pandas nullable boolean |
| Floating annotations | NumPy float32 or float64 as declared |
| Text | pandas string types, including Arrow-backed strings |
| Regions and semantic categories | pandas categorical |

Floating nulls become NaN only where the reader explicitly defines them as
equivalent missing annotations. Otherwise an unsupported conversion fails.
The readers do not change process-global pandas or AnnData settings. Default
nullable-string persistence requires AnnData 0.13 or later and pandas 3.0 or
later, reflected in the package requirements.

This final storage conversion is temporary: general Arrow-backed pandas
numeric columns currently lack AnnData serialization support. It does not
alter expression values or reinterpret identifiers. Support for those columns
is tracked in [AnnData issue #948](https://github.com/scverse/anndata/issues/948).

## Expression and linkage

Expression is returned as eager, finite, canonical SciPy CSR sparse arrays.
Feature and observation order follow the reader's documented format contract;
expression is not densified to validate or link annotations.

Tables are linked only after their final identifiers, regions, and spatial
targets have been constructed. Instance identifiers are non-null, and each
region/instance pair is unique. When an independent target index is available,
table membership must match it exactly. Lazy label images are not computed
merely to enumerate identifiers.

Large transcript points use lazy Dask DataFrames and their Parquet persistence
path. Shapes use GeoDataFrames. These products do not pass through AnnData's
annotation conversion.
