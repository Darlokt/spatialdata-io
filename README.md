![SpatialData banner](https://raw.githubusercontent.com/scverse/spatialdata/main/docs/_static/img/spatialdata_horizontal.svg)

# spatialdata-io: convenient readers for loading common formats into SpatialData

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]
[![DOI](https://zenodo.org/badge/544045123.svg)](https://zenodo.org/badge/latestdoi/544045123)
[![PyPI][badge-pypi]][link-pypi]
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/spatialdata-io/badges/version.svg)](https://anaconda.org/conda-forge/spatialdata-io)

[badge-tests]: https://github.com/scverse/spatialdata-io/actions/workflows/test.yaml/badge.svg
[link-tests]: https://github.com/scverse/spatialdata-io/actions/workflows/test.yaml
[badge-docs]: https://img.shields.io/readthedocs/spatialdata-io
[badge-pypi]: https://badge.fury.io/py/spatialdata-io.svg
[link-pypi]: https://pypi.org/project/spatialdata-io/

spatialdata-io loads spatial omics exports into [SpatialData](https://spatialdata.scverse.org/).
Use it to keep images, labels, transcripts, shapes and expression tables together
in a Zarr store. The [reader guides](docs/readers/index.md) describe supported
inputs, versions, output elements and coordinate units for each technology.

Readers cover Visium, Visium HD, Xenium, CODEX, CosMx, Curio Seeker, DBiT-seq,
MACSima, MCMICRO, MERSCOPE, seqFISH, Steinbock and Stereo-seq. There is also
an experimental ISS reader and support for individual image and GeoJSON files.
Third-party readers also produce SpatialData objects. Sopa supports imaging
platforms including PhenoCycler, MACSima and Hyperion. For mass spectrometry
imaging, METASPACE provides a converter and Thyra supports ion mobility and
MS/MS data from Bruker, Waters, PHI and imzML.
[Sopa](https://github.com/gustaveroussy/sopa), [METASPACE converter](https://github.com/metaspace2020/metaspace-converter), [Thyra](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra)

## Get started

Install the published package in a Python 3.12 or newer environment:

```bash
python -m pip install spatialdata-io
```

From a supported Xenium output directory, for example:

```python
from spatialdata import read_zarr

from spatialdata_io import xenium

sdata = xenium("raw_data")
sdata.write("sample.zarr")
reopened = read_zarr("sample.zarr")
```

Follow [Convert your first dataset](docs/user/first-conversion.md) to download
a small public Xenium experiment and verify its counts after writing. An
[offline image tutorial](docs/user/first-image.md) creates its own TIFF. The published
documentation includes the [Python API][link-api] and [command-line reference][link-cli].
See the [changelog](CHANGELOG.md) when upgrading.

## Help and contributions

Ask usage questions on [scverse Discourse][scverse-discourse] and report bugs
in the [issue tracker][issue-tracker]. For a data-format issue, include the
producer version, directory layout and the output you expected. To add or
extend a reader, see the [contributing guide](docs/contributing.md).

The community maintains these readers independently of the technology vendors.

## Citation

Marconato, L., Palla, G., Yamauchi, K.A. et al. SpatialData: an open and universal data framework for spatial omics. Nat Methods (2024). https://doi.org/10.1038/s41592-024-02212-x

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/scverse/spatialdata-io/issues
[link-docs]: https://spatialdata.scverse.org/projects/io/en/stable/
[link-api]: https://spatialdata.scverse.org/projects/io/en/stable/api.html
[link-cli]: https://spatialdata.scverse.org/projects/io/en/stable/cli.html
[//]: # "numfocus-fiscal-sponsor-attribution"

spatialdata-io is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>
