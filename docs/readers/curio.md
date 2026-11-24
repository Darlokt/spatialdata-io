# Curio Seeker

{func}`spatialdata_io.curio` reads one sample produced by the Curio Seeker
Primary Analysis Pipeline. The export has no pipeline version marker.
Trekker, merged or reprocessed samples, and the uppercase `X_SPATIAL`
coordinate profile are outside the supported layout.

## Dataset layout

All five files are required, with exactly one file per suffix and the same
nonempty sample prefix. Additional files are ignored.

```text
sample/
├── sample_anndata.h5ad
├── sample_cluster_assignment.txt
├── sample_Metrics.csv
├── sample_variable_features_clusters.txt
└── sample_variable_features_spatial_moransi.txt
```

| File | Contract |
| --- | --- |
| H5AD | Expression in `X`, unique nonempty bead and feature IDs, finite real XY coordinates in `obsm["X_spatial"]`. The optional source `obsm["spatial"]` must agree exactly. |
| Cluster assignments | Headerless TSV with bead ID and nonempty cluster label in that order, one row per H5AD bead. |
| Metrics | CSV with metric name and value as the first two columns, then `Category`. Metric names are unique within each category. Additional columns are ignored. |
| Cluster-defining features | TSV with named columns, kept in full. Empty fields become missing values. |
| Moran statistics | TSV with unique known feature IDs in the first column, whose header may be empty. Remaining columns are finite float64 statistics. |

Named headers occupy one physical line and have unique, nonempty names except
for the optional empty Moran ID header. Records match the full header width.
Quoted data fields may span lines. Text values preserve their spelling,
including leading zeros, `NA` and `TRUE`. NUL characters are rejected.
Moran statistics must be present, and numbers round directly to float64.
Spaces and tabs around a number are accepted. Hexadecimal, booleans and digit
separators are rejected. Parsed `nan` and `inf` fail the finiteness check.

## Output and coordinates

- `shapes["cells"]` contains one circle per bead, indexed by the original
  bead ID. The circle is centred at the source XY position and has the
  nominal 5 µm radius (about 7.73 source pixels). The Seeker calibration
  of 0.647 µm per source pixel maps the circles to `global` in micrometres.
- `tables["table"]` preserves H5AD bead and feature order. Categorical
  `obs["region"]` and bead IDs in `obs["instance_id"]` link rows to the
  circles. Cluster labels align by bead ID and their categories follow
  first appearance in table order.
- `X` is a finite canonical CSR sparse array. Duplicate sparse entries
  are summed with a warning. Integer sums that overflow their source dtype
  promote to float32 or float64 when all the resulting values are exact,
  and fail otherwise. Existing layers, `raw`, embeddings and annotations
  are retained.
- `obsm["spatial_pixels"]` preserves the original XY array, and
  `obsm["spatial"]` holds float64 XY micrometres. Both arrays are declared in
  `uns["spatialdata_io_positions"]`. No image is registered or returned.
- Moran statistics are added to `var` by feature ID, with NaN for unlisted
  features. `uns["Top_cluster_defining_features"]` holds the defining-features
  DataFrame, and one `uns[category]` dictionary stores the metrics per
  category. Conflicting enrichments raise. Invalid SpatialData annotation
  names are sanitized with a warning, including case-insensitive collisions.

The result is eager and source files are unchanged. Malformed artifacts and
disagreements raise a contextual `ReaderFormatError`. Discovered paths use
the [dataset path policy](../user/paths.md).

## Python and command line

```python
from spatialdata_io import curio

sdata = curio("sample")
sdata.write("sample.zarr")
```

```bash
python -m spatialdata_io curio --input sample --output sample.zarr
```
