# Reader guides

Choose the guide for the format that produced your data. Each guide describes
the expected files, returned elements and coordinate systems. See the [Python
API](../api.md) for signatures and the [command-line reference](../cli.md) for options.

## Main readers

Use the instrument or pipeline that produced your files to choose a reader.

| Technology or pipeline | Input to look for | Tested versions or export layouts |
| --- | --- | --- |
| [CODEX](codex.md) | Processor cell tables and stitched marker TIFFs, or processed CODEX QPTIFF scans | Regional and montage layouts. Calibration JSON tested with 1.5.0.38 and 1.7.0.6 |
| [CosMx](cosmx.md) | Flat-file export with expression and cell metadata CSVs | 2021 SMI and AtoMx flat-file layouts |
| [Curio Seeker](curio.md) | Primary pipeline H5AD with cluster, metric and feature files | Unversioned Primary Analysis Pipeline export |
| [DBiT-seq](dbit.md) | H5AD and two-axis barcode map | Unversioned barcode-grid export |
| [MACSima](macsima.md) | MACS iQ View processed channel TIFFs | MACSiQView major 0 and 1 (tested with 0.15.3 and 1.1.1) |
| [MCMICRO](mcmicro.md) | MCQuant tables and `qc/params.yml` | Whole-slide and Coreograph tissue-microarray layouts |
| [MERSCOPE](merscope.md) | Vizgen region directory, optionally with a VPT analysis | Instrument output and VPT Cellpose, Cellpose2 and Watershed layouts |
| [PhenoCycler](phenocycler.md) | Whole-slide QPTIFF or component TIFF | Fusion 1.0.5 and 2.3.1, CODEX Instrument Manager, Vectra Polaris 1.1, PhenoImager HT 2.2 and inForm 2.2.1 |
| [seqFISH](seqfish.md) | Spatial Genomics per-ROI tables and images | Unversioned Spatial Genomics ROI export |
| [Steinbock](steinbock.md) | Matched OME images, segmentation masks and an H5AD table | Unversioned Steinbock project |
| [Stereo-seq](stereoseq.md) | SAW or flat exports with selected GEF, H5AD or image products | SAW 7.0/7.1 numbered and 8.0/8.1/8.2 outs layouts, compact numbered and flat exports |
| [Visium](visium.md) | Classic Space Ranger counts and spot positions | Space Ranger 1.x headerless and 2.0+ headered positions |
| [Visium HD](visium_hd.md) | Space Ranger HD binned or segmented outputs | Space Ranger 3.0, 3.1, 4.0 and 4.1 |
| [Xenium](xenium.md) | XOA or Xenium Ranger output with `experiment.xenium` | XOA 1.0–4.x and Xenium Ranger outputs in the corresponding bundle families |

```{toctree}
:hidden:
:maxdepth: 1

codex
cosmx
curio
dbit
macsima
mcmicro
merscope
phenocycler
seqfish
steinbock
stereoseq
visium
visium_hd
xenium
```

## Individual image and shape files

Use the [generic reader](generic.md) for a single TIFF, BTF, PNG, JPEG or polygonal
GeoJSON file.

```{toctree}
:hidden:
:maxdepth: 1

generic
```

## Experimental readers

[In situ sequencing (ISS)](iss.md) reads a caller-selected image, segmentation
and expression table. Import it from `spatialdata_io.experimental`.

```{toctree}
:hidden:
:maxdepth: 1

iss
```
