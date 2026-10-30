# MCMICRO

{func}`spatialdata_io.mcmicro` reads the outputs of one [MCMICRO](https://mcmicro.org/) whole-slide or tissue-microarray (TMA) run. It returns the registered image, Coreograph cores and tissue masks, segmentation labels and one linked MCQuant cell table. Pipeline inputs (`raw/`, `illumination/`), Coreograph QC files and segmentation puncta tables are not read.

## Run layouts

MCMICRO writes `qc/params.yml` as a complete dump of the parameters used. Its YAML boolean `workflow.tma` selects the layout; any other value, a duplicated key or malformed YAML fails. A whole-slide run contains:

```text
run/
├── markers.csv
├── qc/params.yml
├── registration/<image>.ome.tif
├── segmentation/<method>-<image>/<mask>.ome.tif
└── quantification/<image>--<method>_<mask>.csv
```

A TMA run adds numbered Coreograph cores, each with one tissue mask:

```text
run/
├── markers.csv
├── qc/params.yml
├── registration/<tma>.ome.tif
├── dearray/<core>.ome.tif
├── dearray/masks/<core>_mask.tif
├── segmentation/<method>-<core>/<mask>.ome.tif
└── quantification/<core>--<method>_<mask>.csv
```

Registration must contain exactly one TIFF-family image; its image ID drops the extension and an `.ome` suffix. Cores are positive integers and are ordered numerically. A mask stem is the label file name up to its first dot, as MCQuant names it, so `cell_mask_ring.ome.tif` has the stem `cell_mask_ring`. Every quantification table must name exactly one discovered label: `<image>--<method>_<mask>.csv` is compared with whole method and mask names, so underscores are never split. Labels without a quantification table are returned without table rows.

A historical `segmentation/<image>/` directory, without the method prefix, is accepted when exactly one quantification table identifies the method of each of its masks. Segmentation and quantification are optional. All element names must be unique; a layout whose names would collide fails before any image is opened.

## Elements and coordinates

| Element | Contract |
| --- | --- |
| `images["<image>_image"]` | Whole-slide registered image in `global`. |
| `images["tma_map"]` | TMA registered image in `global`. |
| `images["core_<n>_image"]` | Coreograph core in its local frame `core_<n>`. |
| `labels["core_<n>_mask"]` | Coreograph tissue mask in its own frame `core_<n>_mask`. |
| `labels["<image>_<method>_<mask>"]` | Segmentation labels in the frame of their image (`core_<n>` for cores). |
| `tables["table"]` | One table linked to every label with quantified cells. |

Images are lazy `(c, y, x)` arrays in source dtype. Their channels are the canonical MCQuant marker IDs in `markers.csv` row order: every occurrence of a repeated marker is suffixed, so `DAPI, CD3, DAPI` becomes `DAPI_1, CD3, DAPI_2`, and generated collisions such as `X, X, X_1` fail. The registered image and every core must have exactly one channel per marker. Labels are lazy integer `(y, x)` arrays that must match the spatial shape of their source image; masks keep their own, possibly downsampled, grid.

Each image uses complete base OME X/Y physical sizes in µm, nm or mm to scale its local frame to micrometres. Missing or incomplete sizes retain the entire group in source pixels. Present invalid sizes or unsupported units fail. Anisotropic spacing and the native origin are preserved. Same-grid segmentations inherit the image calibration when their own declaration is absent; conflicting declared sizes fail. Independent tissue masks use only their own complete OME metadata. Coreograph exports no authoritative crop origins and masks may be downsampled, so no registration between map, cores and tissue masks is inferred. ASHLAR may write its own fallback spacing when acquisition metadata was unavailable: declared OME sizes are used as source metadata, not certified acquisition measurements.

## Table

`markers.csv` needs a `marker_name` column. Every other column is optional module metadata and is kept as exact text in `var`, together with the original `marker_name`. `var_names` are the canonical marker IDs.

Quantification tables need `CellID`, one column per canonical marker and `X_centroid`/`Y_centroid`. All tables of a run must have the same columns. Marker columns become eager, finite, canonical CSR float64 expression; exact decimal values are decoded with correct rounding, and nonfinite values or nonzero values that underflow to zero fail. `CellID` holds exact positive int64 label values, unique within each label and representable in that label's dtype; label pixels are never read to check them. MCQuant morphology columns (`Area`, `MajorAxisLength`, `MinorAxisLength`, `Eccentricity`, `Solidity`, `Extent`, `Orientation`, `column_centroid`, `row_centroid`) are float64 with empty fields as missing values. All other columns, including suffixed intensity properties, are kept as exact text.

Rows follow label order and then source row order. Observation names are row positions `"0"`, `"1"`, ...; `region` names the label image and `CellID` is the instance key. `obsm["spatial_pixels"]` preserves the unchanged float64 centroid XY values. MCQuant obtains these directly from regionprops zero-based pixel-index centres. Their native frames are named `<source-image-frame>_regionprops_indices`, distinct from the raster's pixel-corner frame.

When every source image represented by measured rows has complete XY calibration, `obsm["spatial"]` contains float64 XY micrometres in the corresponding raster frame, calculated as `(source_xy + 0.5) * spacing_xy_um`. For example, the source single-pixel centroid `(2, 1)` at spacing `(2, 3)` µm/pixel becomes `(5, 4.5)` µm. Several segmentations of one image share that frame. Independent cores remain independent, with categorical `obs["spatial_frame"]`; native positions use categorical `obs["spatial_pixel_frame"]` when multiple frames occur. A single represented frame is declared directly without a constant observation column. Missing, partial or mixed calibration omits physical `spatial` for the whole aggregate, retaining every row and native position. Unmeasured images and unrelated tissue masks do not decide table calibration; segmentation metadata cannot complete a partially calibrated source image.

`uns["spatialdata_io_positions"]` records version-1 position descriptions, units, frames and calibration evidence. Existing columns that collide with generated frame columns fail. Positions are snapshots: later element registration and ordinary AnnData concatenation do not update or register these arrays. Re-export positions explicitly after registration.

Area retains pixel², MajorAxisLength/MinorAxisLength and column_centroid/row_centroid retain pixels, and Eccentricity/Solidity/Extent are dimensionless. Orientation retains radians from the row axis to the major axis, counter-clockwise in the range −π/2 to π/2, following [MCQuant’s direct regionprops mapping](https://github.com/labsyspharm/quantification/blob/61a1f539665864d7a2896c6754d90de404c16759/mcquant/SingleCellDataExtraction.py#L47-L76) and [scikit-image’s convention](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops); concatenating cores does not register their independent physical frames. A run without quantification returns no table.

## Options and execution

`scale_factors` and `chunks` follow the shared [raster options](raster_options.md): by default a single scale, images in `(1, 1024, 1024)` chunks and labels in `(1024, 1024)` chunks. Generated label pyramids require labels of at most 32 bits; 64-bit labels need the default single scale.

Construction reads TIFF metadata only and executes no pixel tasks. Computing a region reads only the TIFF tiles, or strips grouped into bands of at most 16 MiB, that intersect its chunks. Quantification tables are parsed in bounded batches of at most 10,000 rows and 8 MiB of dense marker values, and only CSR blocks are retained. The source files must remain available for later raster computation.

## Python and command line

```python
from spatialdata_io import mcmicro

sdata = mcmicro("run", scale_factors=(2, 2))
sdata.write("run.zarr")
```

The CLI mirrors the reader with repeatable pyramid and chunk options:

```bash
spatialdata_io mcmicro --input run --output run.zarr --scale-factors 2 --scale-factors 2
```

The former `imread_kwargs`, `image_models_kwargs` and `labels_models_kwargs` mappings have been removed; use `scale_factors` and `chunks`. The default chunks are `(1, 1024, 1024)` for images and `(1024, 1024)` for labels instead of storage-aligned chunks, and `scale_factors=()` replaces `None` for a single scale. Compared with earlier releases, the reader no longer returns raw or illumination images, includes the method in label names, keeps every core and mask in its exact local frame instead of placing it by centroids, names masks `core_<n>_mask`, returns one table named `table` with positional observation names, and stores expression as sparse arrays. Native pixel centroids now use `spatial_pixels`; canonical `spatial` is reserved for completely calibrated physical positions.
