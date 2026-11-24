# How the Visium reader works

This classic Visium input uses the named positions layout and an HDF5 count
matrix. The hires PNG is optional. [Visium input and output guide](../readers/visium.md)

```text
outs/
├── filtered_feature_bc_matrix.h5
└── spatial/
    ├── tissue_positions.csv
    ├── scalefactors_json.json
    └── tissue_hires_image.png
```

Follow the calls in `src/spatialdata_io/readers/visium/` to see where each
decision belongs.

1. **Identify the input.** `_reader.visium()` validates options and calls
   `_layout.discover_layout()` once. The layout chooses `H5Counts` with
   the validated matrix path and its `CountMetadata`. The MEX alternative
   uses `MtxCounts` with all three validated member paths. Both feed one
   `VisiumLayout`, with a `PositionFormat` classified from the first logical
   CSV record and optional image paths. The path utilities enforce kinds and
   containment. Visium owns the HDF5-before-MEX precedence and determines
   which filenames and metadata are meaningful.
2. **Read the images and establish frames.** `_metadata.read_scale_factors()`
   uses the shared JSON reader, then checks the three positive vendor scale
   fields. `_geometry.make_frames()` establishes the full-resolution and
   downsampled pixel frames. `_images.read_images()` uses the shared PNG
   loader and registers the hires image into the full-resolution frame with
   the reciprocal of the vendor scale. A hires scale of `0.5` maps one hires
   pixel to two full-resolution pixels. These are display scales, so the
   reader exposes pixel frames.
3. **Read and align expression and positions.** `_tables.read_positions()`
   projects the six fields declared by `_schemas.POSITION_SOURCE_SCHEMA`
   with Arrow. `_schemas.normalize_positions()` checks values and indexes
   rows by barcode. `_tables.prepare_table()` reads sparse counts through
   `read_10x_h5()` or `read_10x_mtx()`, then aligns the position rows to the
   count matrix by exact barcode, preserving matrix order. It stores the
   vendor's column and row values as XY in `obsm["spatial_pixels"]` and
   declares their units and frame.
4. **Build and link the targets.** `_geometry.make_circles()` builds one
   circle per final row using the spot display diameter divided by two.
   `_tables.link_table()` prepares AnnData annotations and calls
   `parse_linked_table()` with `(region, spot_id)` and the circle index.
   `visium()` assembles `tables["table"]`, `shapes[dataset_id]` and any
   selected images. It records reader, package and available vendor version
   provenance in `SpatialData.attrs` before returning.

The Click command reads the same layout and writes the resulting SpatialData
to Zarr. Supply `--dataset-id` to give the output a known name.

```bash
uv run --no-sync python -m spatialdata_io visium --input outs --output sample.zarr --dataset-id sample
```

In the tests, `tests/unit/readers/visium/test_layout.py` checks path
precedence and ambiguous alternatives, `test_schemas.py` checks typed
positions, and `test_tables.py` checks alignment and linkage. The synthetic
integration test `test_exact_scientific_contract` in
`tests/integration/readers/visium/test_visium.py` exercises HDF5 and MEX with
different position layouts and checks the raster, spot coordinates and
linked expression together. The command tests in
`tests/unit/cli/commands/test_visium.py` check how CLI values reach the
reader.


To apply this design to a new format, follow [Add or update a reader](readers.md).
