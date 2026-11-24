# spatialdata-io

```{image} _static/img/spatialdata_horizontal.svg
:alt: SpatialData
:class: dark-light
```

Load spatial omics data into a SpatialData object or write it to a Zarr store.
Choose the reader for your instrument or pipeline to keep images, segmentations,
transcripts and expression tables together in their documented coordinate systems.

- **New to spatialdata-io?** [Convert your first dataset](user/first-conversion.md) using a small, public Xenium experiment. For an offline example, [convert a single image](user/first-image.md).
- **Working with an experiment?** [Find your reader](readers/index.md) by technology. Each guide lists the expected files, supported options and output elements.
- **Choosing coordinates for analysis?** [Understand frames and units](user/coordinates.md) before comparing positions across images or samples.
- **Adding support for a format?** Start with the [contributing guide](contributing.md).

For analysis and visualization after conversion, see the [SpatialData documentation](https://spatialdata.scverse.org/en/stable/).
spatialdata-io belongs to the scverse ecosystem {cite:p}`Virshup_2023`, with fiscal sponsorship from NumFOCUS. [scverse governance](https://scverse.org/about/roles), [NumFOCUS](https://numfocus.org/project/scverse)

If you use SpatialData in your work, please cite {cite:t}`marconatoSpatialDataOpenUniversal2024`.

```{toctree}
:caption: Getting started
:hidden:
:maxdepth: 1

user/first-conversion
user/first-image
```

```{toctree}
:caption: How-to guides
:hidden:
:maxdepth: 1

user/rasters
user/paths
user/legacy_anndata
```

```{toctree}
:caption: Understanding reader output
:hidden:
:maxdepth: 1

user/coordinates
user/tables
```

```{toctree}
:caption: Reference
:hidden:
:maxdepth: 2

readers/index
api
cli
```

```{toctree}
:caption: Project
:hidden:
:maxdepth: 2

development/index
changelog
references
```
