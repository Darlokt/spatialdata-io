# PhenoCycler and Akoya QPTIFF

{func}`spatialdata_io.phenocycler` opens a processed PhenoCycler-Fusion scan,
a CODEX Instrument Manager QPTIFF, a Vectra or PhenoImager quantitative pathology
scan, or an inForm component image. Pass a QPTIFF file or a directory of
processed QPTIFFs. For a component image with a `.tif` extension, pass the
file itself. For CODEX Processor cell tables and stitched marker planes, see
[CODEX](codex.md).

```text
scans/
├── Tonsil_Scan1.qptiff
├── Tonsil_Scan1.er.qptiff     # optional extended-range version
└── Tonsil_Scan1.xpd           # optional experiment protocol
```

The HuBMAP deposit layout separates final scans from per-cycle acquisition
intermediates and channel annotations:

```text
dataset/
├── raw/images/
│   ├── Sample.qptiff
│   └── phenocycler/           # acquisition intermediates
└── lab_processed/images/
    └── Sample.qptiff.channels.csv
```

[HuBMAP PhenoCycler file organization](https://docs.hubmapconsortium.org/doi-pages/phenocycler)

Directory reads prefer the extended-range `.er.qptiff` when both encodings
of a scan are present. Pass the ordinary file directly to select it. Set
`read_all_bit_depths=True` to also read the matching 8-bit companion. For
`Tonsil_Scan1`, the two images are `Tonsil_image` and `Tonsil_8bit_image`.
Use `regions=["Tonsil"]` for a subset of a directory. The reader preserves
distinct scan suffixes when stripping them would merge scans.

## Images and measurements

The reader returns one lazy `(c, y, x)` image per selected scan region. The
XML `ImageType` of each TIFF page identifies the full-resolution measurements.
Thumbnails, slide labels and overview images are excluded. Fluorescence bands
stay in source order and preserve their stored dtype. Brightfield RGB produces
red, green and blue channels. inForm component planes can contain float32
values. The source files must remain available when computing or writing
images. `chunks` sets output chunk sizes and `scale_factors` requests generated
pyramid levels. [Raster options](../user/rasters.md)

A fluorescence page can identify both a target (`Biomarker`, e.g. NPM1C) and
an acquisition band (`Name`, e.g. ATTO 550). Quantitative attributes such as
`SignalUnits` and `ScaleFactor` describe the stored samples. The reader
preserves the pixel values. An 8-bit display image and its extended-range
counterpart can therefore have different numerical values.
[Akoya QPTIFF specification, Appendix B](https://www.quanterix.com/wp-content/uploads/2026/01/Phenochart-User-Guide-for-version-2.2.0.pdf)

For channel metadata in array order, inspect
`sdata.attrs["akoya"]["regions"][region]["channels"]`. Each entry identifies
the source TIFF page, its source-derived ID, and the displayed channel name.
Available fields include `biomarker`, `band_name`, `signal_units_code`, and
`scale_factor`. Repeated marker names receive distinct channel coordinates.
Setting `channel_names` changes display names while retaining source identity.
XPD and `Marker`-column CSV summaries report marker agreement with the scan.
A HuBMAP channel CSV instead records one row per QPTIFF channel, with its
source `channel id`, two segmentation-use flags and an antibody flag. These
rows are available under `channels_csv["annotations"]` in the region metadata,
in submitted order. Their IDs can use a different convention from the QPI
marker names. Segmentation-use flags describe how channels were used in
analysis. The reader returns images, and the flags themselves supply no cell
boundaries. Extra CSV columns, raw image descriptions and operator records
are excluded from the output metadata.
[HuBMAP channel definition](https://docs.google.com/spreadsheets/d/1JikzRNyDErspgPSel4P9Y6gx6N_mrAGIMDcJZd8TucA/edit#gid=0)

## Physical coordinates

Akoya records a known physical pixel size using equal X and Y resolutions
in pixels per centimetre. For example, 20,000 pixels/cm corresponds to
0.5 µm/pixel. The 96 pixels/inch convention means calibration is unknown.
When present with calibrated pixels, the stage position locates the upper-left
corner of the full-resolution image in micrometres. Pixel centres are half a
pixel inside that corner. For a single calibrated scan the frame is `global`.
Several scans can share `global` when the slide identifier matches and every
scan has calibrated pixels and a stage position. This preserves a field's
location when selecting a subset. Independent calibrated fields have regional
micrometre frames. With unknown calibration, a single scan uses `image_pixels`
and multiple scans use `<region>_pixels`. The nominal scan-profile resolution,
when available, is retained as acquisition metadata.
[Akoya’s field geometry implementation](https://github.com/akoyabio/phenoptr/blob/7d5dceaacdc1af3e2ff595bf0aeb09172fd546c2/R/read_components.R#L102-L123)

## Read a region

```python
from spatialdata_io import phenocycler

sdata = phenocycler("scans/Tonsil_Scan1.qptiff", chunks=(1, 512, 512))
image = sdata.images["Tonsil_image"]
spot = image.isel(y=slice(1000, 1020), x=slice(400, 420)).compute()
```

The equivalent CLI command writes the result to a SpatialData Zarr store:

```bash
python -m spatialdata_io phenocycler -i scans/Tonsil_Scan1.qptiff -o tonsil.zarr
```

Two public examples illustrate the difference between marker and band names.
The Fusion 1 `AML2_183_Scan1.qptiff` has 54 measurement channels. Its second
channel records NPM1C measured with ATTO 550. The Fusion 2 Opal
`Tonsil-ctrl_Scan1.qptiff` identifies its six bands by acquisition names,
including DAPI and Opal 690.
[Acute myeloid leukemia and negative staging marrow scans](https://doi.org/10.25452/figshare.plus.25123538)
and [BioImage Archive study S-BIAD3467](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD3467).
