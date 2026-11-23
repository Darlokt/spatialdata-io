# 10x counts

The shared reader returns an owned, eager `AnnData` with barcodes as observations
and all features in source order. `X` is a canonical CSR sparse array. Integer
counts use float32 through 2²⁴ and float64 through 2⁵³. Larger integer counts
raise. Floating HDF5 counts must already be finite, nonnegative and integral.
The reader closes every source before returning.

- `read_10x_h5()` takes a validated file. It reads a single-genome legacy
  group or a modern `matrix` group, including probe IDs and feature annotations.
  Feature-by-barcode CSC storage supplies the observation-by-feature CSR
  buffers directly. Duplicate entries, misaligned annotations and pointers that
  leave counts unreferenced are malformed. Required datasets must be stored
  inside the file. `read_10x_h5_components()` returns the same owned CSR and
  annotation frames for consumers such as HD nucleus aggregation that use the
  raw matrix as an intermediate. Both entry points skip nested metadata groups.
- `read_10x_mtx()` takes the three validated paths in `MtxFiles`. The layout
  declares gzip compression and whether the feature TSV has the two legacy
  fields or the modern feature-type field. Barcode, feature ID and symbol
  fields are literal UTF-8 strings, including quotes, leading zeros
  and null-like text. The optional `make_unique` changes repeated feature names
  using AnnData's naming convention. The reader supports general and
  lower-triangular symmetric integer coordinate matrices. It checks record
  count, one-based coordinates, decimal integer spelling and nonnegative
  counts before combining repeated coordinates. Arrow
  parses complete lines from 4 MiB source blocks. A single record must fit in
  one source block. The returned sparse array is eager.

`read_10x_h5_axes()` and `read_10x_mtx_fields()` give Xenium transcript-only
reads the feature vocabulary and barcodes without decoding expression.

Readers own discovery, feature selection, scaling, aggregation and linkage.
