# Reader guides

Choose the guide for the format that produced your data. Each guide describes
the expected files, returned elements and coordinate systems. See the [Python
API](../api.md) for signatures and the [command-line reference](../cli.md) for options.

## Main readers

Use the instrument or pipeline that produced your files to choose a reader.

| Technology or pipeline | Input to look for |
| --- | --- |
| [CODEX / PhenoCycler](codex.md) | One regional processor CSV or FCS cell table and an optional stitched image |
| [CosMx](cosmx.md) | Flat-file export with expression and cell metadata CSVs |
| [Curio Seeker](curio.md) | Primary pipeline H5AD with cluster, metric and feature files |
| [DBiT-seq](dbit.md) | H5AD and two-axis barcode map |
| [MACSima](macsima.md) | MACS iQ View processed channel TIFFs |
| [MCMICRO](mcmicro.md) | Whole-slide or tissue-microarray run with `qc/params.yml` |
| [MERSCOPE](merscope.md) | Vizgen region directory, optionally with a VPT analysis |
| [seqFISH](seqfish.md) | Spatial Genomics per-ROI tables and images |
| [Steinbock](steinbock.md) | Matched OME images, segmentation masks and an H5AD table |
| [Stereo-seq](stereoseq.md) | SAW or flat exports with selected GEF, H5AD or image products |
| [Visium](visium.md) | Classic Space Ranger counts and spot positions |
| [Visium HD](visium_hd.md) | Space Ranger HD binned or segmented outputs |
| [Xenium](xenium.md) | XOA or Xenium Ranger output with `experiment.xenium` |

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
