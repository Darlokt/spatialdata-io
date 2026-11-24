# Contributing guide

This document aims at summarizing the most important information for getting you started on contributing to this project.
We assume that you are already familiar with git and with making pull requests on GitHub.

For more extensive tutorials, that also cover the absolute basics,
please refer to other resources such as the [pyopensci tutorials][],
the [scientific Python tutorials][], or the [scanpy developer guide][].

[pyopensci tutorials]: https://www.pyopensci.org/learn.html
[scientific Python tutorials]: https://learn.scientific-python.org/development/tutorials/
[scanpy developer guide]: https://scanpy.scverse.org/page/dev/

For a reader or converter contribution, see [Contributing readers and converters](#contributing-readers-and-converters), including [public data and permissive licenses](#the-importance-of-public-data-and-permissive-licenses) and the [submission checklist](#wrap-up-checklist).

This package is part of scverse and shares the SpatialData development stack. Use uv for a local checkout. Hatch also manages isolated documentation environments and the Python test matrix used by CI. You can use pip in your own virtual environment. [scverse project and governance](https://scverse.org/about/roles), [SpatialData contribution guide](https://github.com/scverse/spatialdata/blob/main/docs/contributing.md), [scverse project template](https://github.com/scverse/cookiecutter-scverse)

[hatch]: https://hatch.pypa.io/latest/

(set-up-a-development-checkout)=

## Installing dev dependencies

Clone your fork and change into its root directory. Use Python 3.12 or newer.
In addition to the packages needed to _use_ this package,
you need additional python packages to [run tests](#writing-tests) and [build the documentation](#docs-building).

:::::{tab-set}
::::{tab-item} uv
:sync: uv

[uv][] creates a project environment in `.venv`. Install the development and test dependency groups and select `.venv` as your interpreter in VS Code or your preferred editor. Include the `doc` group when building the documentation.

```bash
uv sync --group dev --group test
uv run --no-sync python -m spatialdata_io --help
uv run --no-sync python -c "import sys; print(sys.executable)"
```

::::

::::{tab-item} Hatch
:sync: hatch

[Hatch][] creates the development and test environments specified in `pyproject.toml`. To set up the checkout and locate the test environment for your editor, run:

```bash
hatch env create
hatch run python -m spatialdata_io --help
hatch env show -i
hatch env create hatch-test.py3.14-stable
hatch env find hatch-test.py3.14-stable
```

Select the returned environment's Python interpreter in your IDE. In VS Code, the [Hatch extension][hatch-code] lists environments under **Python Environment Managers**.

::::

::::{tab-item} Pip
:sync: pip

Create a virtual environment and install the checkout in editable mode. pip 25.1 and newer can install the dependency groups from `pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . --group dev --group test --group doc
```

Select `.venv` in your editor. On Windows, use `.venv\Scripts\activate` in Command Prompt.

::::
:::::

[hatch environments]: https://hatch.pypa.io/latest/tutorials/environment/basic-usage/
[hatch-code]: https://marketplace.visualstudio.com/items?itemName=PyPA.hatch
[uv]: https://docs.astral.sh/uv/

## Code-style

This package uses [pre-commit][]-style hooks to enforce consistent code-styles.
Run them with [prek][], which reads `.pre-commit-config.yaml` and provides built-in file checks.
On every commit, the checks will either automatically fix issues with the code, or raise an error message.

The hooks run the tools installed in the repository's `.venv`. Create or refresh that environment before installing or running the hooks, including when invoking them through Hatch. Review any fixes before committing.

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv sync --group dev --group test
uv run --no-sync prek install
uv run --no-sync prek run --all-files
```
::::

::::{tab-item} Hatch
:sync: hatch

```bash
hatch run uv sync --no-active --group dev --group test
hatch run prek install
hatch run prek run --all-files
```
::::
:::::

For read-only checks, run the linter, formatter check and type checker. The configurations for Ruff and Pyrefly live in `pyproject.toml`.

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv run --no-sync ruff check --no-fix .
uv run --no-sync ruff format --check .
uv run --no-sync pyrefly check
```
::::

::::{tab-item} Hatch
:sync: hatch

```bash
hatch check code
hatch check fmt
hatch check types
```
::::
:::::

If you didn’t run the checks locally, the `Pre-commit checks` job of the GitHub Actions CI runs them on your pull request and reports any failures.
We strongly encourage installing and running the checks locally first to understand their usage.

Finally, most editors have an _autoformat on save_ feature.
Consider enabling this option for [Ruff][ruff-editors].
Prek's built-in JSON formatter preserves key order and uses four-space indentation.
JSONC and CSS files are not automatically formatted.

[pre-commit]: https://pre-commit.com/
[prek]: https://prek.j178.dev/
[ruff-editors]: https://docs.astral.sh/ruff/editors/

(writing-tests)=

## Writing tests

This package uses [pytest][] for automated testing.
Write tests for new or changed behavior. See the [reader testing guide](development/testing.md) for fixtures, public datasets and coverage, and {doc}`scanpy:dev/testing` for background on testing scverse code.

Most IDEs integrate with pytest and provide a GUI to run tests.
If you set up your virtual environments as described in [installing dev dependencies](#installing-dev-dependencies),
test cases should be automatically discovered by your IDE.

Alternatively, you can run all tests from the command line by executing

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv run --no-sync pytest
# Run the stable CI versions in isolated environments.
uv run --isolated --python 3.12 --group dev --group test pytest
uv run --isolated --python 3.14 --group dev --group test pytest
```

::::

::::{tab-item} Hatch
:sync: hatch

```bash
hatch test  # run in a compatible test environment
# or
hatch test --all  # test with all supported Python versions
```

::::

::::{tab-item} Pip
:sync: pip

```bash
source .venv/bin/activate
pytest
```

::::
:::::

in the root of the repository.

[pytest]: https://docs.pytest.org/

### Continuous integration

Pull requests run the stable-dependency suite on Python 3.12 and 3.14. An advisory job on Python 3.14 also installs pre-release dependencies to catch incompatibilities before release. The matrix is defined in `pyproject.toml` and run by `.github/workflows/test.yaml`.

## Publishing a release

### Updating the version number

The version number is derived from the git tag by [hatch-vcs][], so there is nothing to edit in `pyproject.toml`.
Add the release notes to `CHANGELOG.md` and pick the tag according to [Semantic Versioning][semver], in brief

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
> 1. MAJOR version when you make incompatible API changes,
> 2. MINOR version when you add functionality in a backwards compatible manner, and
> 3. PATCH version when you make backwards compatible bug fixes.
>
> Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

Build the distributions locally before tagging.

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv build
```
::::

::::{tab-item} Hatch
:sync: hatch

```bash
hatch build
```
::::
:::::

Once you are done, commit and push your changes and navigate to the "Releases" page of this project on GitHub.
Specify `vX.X.X` as a tag name and create a release.
For more information, see [managing GitHub releases][].
This will automatically create a git tag and trigger a Github workflow that creates a release on [PyPI][].

[hatch-vcs]: https://github.com/ofek/hatch-vcs
[semver]: https://semver.org/
[managing GitHub releases]: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
[pypi]: https://pypi.org/

## Writing documentation

Please write documentation for new or changed features and use cases. Put reader input and output contracts in `docs/readers/`, user tasks in `docs/user/`, and contributor workflows in `docs/development/`. The site distinguishes tutorials, how-to guides, reference and explanation. [Diátaxis](https://diataxis.fr/)
This project uses [sphinx][] with the following features:

- The [myst][] extension allows to write documentation in markdown/Markedly Structured Text
- [Numpy-style docstrings][numpydoc] (through the [Napoleon][numpydoc-napoleon] extension).
- Jupyter notebooks as tutorials through [myst-nb][] (See [Tutorials with myst-nb](#tutorials-with-myst-nb-and-jupyter-notebooks))
- [sphinx-autodoc-typehints][], to automatically reference annotated input and output types
- Citations (like {cite:p}`Virshup_2023`) can be included with [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/)

See scanpy’s {doc}`scanpy:dev/documentation` for more information on how to write your own.

[sphinx]: https://www.sphinx-doc.org/
[myst]: https://myst-parser.readthedocs.io/page/intro.html
[myst-nb]: https://myst-nb.readthedocs.io/
[numpydoc-napoleon]: https://www.sphinx-doc.org/page/usage/extensions/napoleon.html
[numpydoc]: https://numpydoc.readthedocs.io/page/format.html
[sphinx-autodoc-typehints]: https://github.com/tox-dev/sphinx-autodoc-typehints

### Tutorials with myst-nb and jupyter notebooks

Use [MyST-NB][myst-nb] for notebook tutorials. Place the notebook under `docs/` and add it to the appropriate toctree.
Currently, only notebooks in `.ipynb` format are supported that will be included with both their input and output cells.
It is your responsibility to update and re-run the notebook whenever necessary.

If you are interested in automatically running notebooks as part of the continuous integration,
please check out [this feature request][issue-render-notebooks] in the `cookiecutter-scverse` repository.

[issue-render-notebooks]: https://github.com/scverse/cookiecutter-scverse/issues/40

#### Hints

- If you refer to objects from other packages, please add an entry to `intersphinx_mapping` in `docs/conf.py`.
  Only if you do so can sphinx automatically create a link to the external documentation.
- If building the documentation fails because of a missing link that is outside your control,
  you can add an entry to the `nitpick_ignore` list in `docs/conf.py`

(docs-building)=

### Building the docs locally

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv sync --group dev --group test --group doc
uv run --no-sync sphinx-build -M html docs docs/_build -W
uv run --no-sync python -m webbrowser -t docs/_build/html/index.html
```

::::

::::{tab-item} Hatch
:sync: hatch

```bash
hatch run docs:build
hatch run docs:open
```

::::

::::{tab-item} Pip
:sync: pip

```bash
source .venv/bin/activate
python -m sphinx -M html docs docs/_build -W
python -m webbrowser -t docs/_build/html/index.html
```

::::
:::::

## Contributing readers and converters

This section is intended for:

- **Spatial technology companies** that would like to add a reader (or converter) for a new product, support a new version of an existing technology, or contribute with edits/bugfixes to one of the available readers.
- **Academic labs** that would like to contribute a reader (or converter) for in-house data generation pipelines.
- **General users** that would like to improve the library.

The `spatialdata-io` repository inherits the technical stack of `spatialdata`. Therefore, please also read the [contribution guide from the `spatialdata` repository](https://github.com/scverse/spatialdata/blob/main/docs/contributing.md).

The remainder of this document is specific to `spatialdata-io`. Feedback on this document is highly welcomed: you are encouraged to open a PR to improve this guide.
**A summary checklist is provided at the end of this document.**

### Adding a new reader

To contribute a reader and make code review efficient, we kindly ask for:

- the **specification** for the raw data: a link to a (ideally) _public_ resource describing how the data is organized, and/or, for a new version of an already supported technology, a changelog describing in detail which changes are introduced;
- the **reader** itself: a Python function, generally supported by helper functions/classes/enums, that reads the raw data and returns a `SpatialData` object;
- example **data**: preferably one or more (very) small _public_ datasets, with a permissive license, to represent the data across various versions and cover potential edge cases of the raw data format. Alternatively (or preferably, in addition), scripts to easily download _public_ and licensed data.
- **test** functions: to ensure that the reader parses the data correctly. If helper functions are used, they should also be tested.

#### The specification

In your issue or PR, please include a link to a _public_ specification and/or changelog describing how the raw data is organized on disk and what changes across various versions.

If the specification cannot (yet) be made public, you can reach out to scverse via [Zulip](https://scverse.zulipchat.com/#narrow/channel/443514-spatialdata-dev) via private message, or via email.

The advantage of a public specification is that it makes contributions and bugfixes from the community easier.

#### Reader architecture

Dataset readers take `path`, discover their layout once, build the requested elements and link their tables. The developer guide explains how to register and test a new reader. A worked example traces a Visium dataset from its files to a linked table and images. [Adding a reader](development/readers.md), [Visium walkthrough](development/visium-walkthrough.md), [Shared utilities](development/contracts.md)

##### Experimental readers

If a specification is not available, a reader has less guarantee to work out-of-the-box. This is, for instance, the case for spatial omics datasets generated with in-house technologies. Even in this case, users may find a reader beneficial: they may still have to adapt the reader for their use cases, but it could provide a good starting point to parse their data. Keep experimental readers under `spatialdata_io.experimental` and document their supported artifact contracts with executable workflows. The experimental [ISS reader](readers/iss.md) reader, for example, supports explicitly selected TIFF images, integer segmentation labels, and a linked h5ad table.

### Adding tests: small data vs real data

An effective way to make a specification easier to understand and to improve the code review process is to provide test data.

#### When small public test datasets are available

Prefer small public datasets (roughly 100 kB–10 MB when practical) that contributors and CI can download and exercise. The preparation workflow fetches registered fixtures. For an example of tests that compare reader output with original vendor files, see `tests/integration/readers/xenium/test_data.py`. [Register and download a test dataset](development/datasets.md), [CI data preparation](https://github.com/scverse/spatialdata-io/blob/main/.github/workflows/prepare_test_data.yaml)

Notes:

- The small test datasets do not need to represent real data and can be entirely artificial.
- If a format specification contains edge cases, we recommend employing a dataset that covers the various edge cases, or using multiple test datasets.
- The data should be licensed with a permissive format, e.g. [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
- If the dataset is not accessible from the web, we invite you to submit it to the ["SpatialData Submissions" Zenodo community](https://zenodo.org/communities/scverse-spatialdata-submissions/) to make it available to the public.

#### When only real (large) public data is available

Even when a small dataset is not available, we encourage you to produce one. If this is not possible, a public real dataset (with a permissive license) can be used. It is also possible to use both: small test datasets for CI testing, and a real dataset for double-checking real-world usability and performance.

To streamline the usage of datasets, we kindly ask you to provide two scripts via a PR to the [`spatialdata-sandbox`](https://github.com/giovp/spatialdata-sandbox) repository (not `spatialdata-io`): a script to download the data and a script to convert the downloaded data to the SpatialData Zarr format.

##### Folder structure

Please create a folder in the [root directory of `spatialdata-sandbox`](https://github.com/giovp/spatialdata-sandbox/tree/main) named after your technology. Please include a format version if available. We also add the suffix `_io`: we use this to distinguish between scripts that reuse `spatialdata_io` for converting the data to SpatialData Zarr and scripts that parse the data from scratch.

A note: `spatialdata-sandbox` is primarily used by scverse developers and, as the name suggests, we mainly use it as a sandbox for experimenting and for small scripts. We do not have pre-commit installed nor CI set up, so feel free not to worry about the code style.

##### The `download.py` script

When the data does not follow any specification (for instance, for in-house data), it sometimes also performs some manipulation on the data and, in such cases, stores the result in any convenient way. Example [`download.py`](https://github.com/giovp/spatialdata-sandbox/blob/main/merfish/download.py) for in-house MERFISH data.

##### The `to_zarr.py` script

The `to_zarr.py` script: (1) imports a suitable reader from `spatialdata_io`; (2) reads the raw data from the folder `data` to obtain a `SpatialData` object; (3) saves the object to disk to a new Zarr store called `data.zarr`; (4) as a consistency check, reads the store back into memory to a new `SpatialData` object.

#### The importance of public data and permissive licenses

There are several reasons to prefer public datasets provided with a permissive license (e.g. 10x public datasets often use the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)):

- It allows the community to use the `download.py` script to download the data. An example of data accessible to the public but not easily downloadable is when the data can be downloaded only after completing a questionnaire. With private data, collaborations and community contributions are difficult.
- It allows the community to modify (e.g. subset, to make download faster) and rehost the data. This makes it easier to share results and collaborate effectively.

#### When only private data is available

Due to the open-source, collaborative nature of our work, developing readers for private datasets is out of our scope. Therefore, we will not be able to consider private datasets. An exception is when developing a reader for a technology whose launch is scheduled and imminent, and for which the data will soon become public. If you find yourself in this scenario, please reach out to scverse via [Zulip](https://scverse.zulipchat.com/#narrow/channel/443514-spatialdata-dev) via private message, or via email. In such cases, we will ask you to give us access to a private repository where you can upload a `download.py` and `to_zarr.py` script working as described above. Alternatively, you can also upload the data privately to the ["SpatialData Submissions" Zenodo community](https://zenodo.org/communities/scverse-spatialdata-submissions/), instead of providing a `download.py` script.

**Technical notes**

If the `download.py` and `to_zarr.py` scripts require Python imports for packages that are not available in an environment where `spatialdata` is installed, please also provide a `requirements.txt` file (see for instance a [legacy one here](https://github.com/giovp/spatialdata-sandbox/blob/main/merfish/requirements.txt)), or specify the dependencies in the scripts as [inline script metadata (PEP 723)](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies).

### What to test

We encourage testing the reader function and any helper function.

Keep fast unit tests under `tests/unit/` and reader, CLI and Zarr workflows under `tests/integration/`. A small deterministic fixture for each supported layout lets anyone run the offline tests.

#### Download and contribute test data

Register public examples by name in `scripts/test_data_downloader/datasets.toml` with provenance, licence and checksums. The downloader verifies files with Pooch and accepts a dataset key or reader group.

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv run --no-sync pytest -m "xenium and not data"
uv run --no-sync python scripts/test_data_downloader --list
uv run --no-sync python scripts/test_data_downloader --dataset xenium_prime_mouse_ileum
uv run --no-sync pytest -m "xenium and data"
```
::::

::::{tab-item} Hatch
:sync: hatch

```bash
hatch test -- -m "xenium and not data"
hatch run python scripts/test_data_downloader --list
hatch run python scripts/test_data_downloader --dataset xenium_prime_mouse_ileum
hatch test -- -m "xenium and data"
```
::::
:::::

Keep the downloaded files under `data/`, or pass `--output` and set `SPATIALDATA_IO_TEST_DATA_DIR` to the same parent for pytest. To contribute a dataset, record its producer, software version and licence, choose a ZIP, DOI or individually checksummed files, then verify the fixture from a clean download. The [dataset guide](development/datasets.md) includes the manifest fields, checksum command and an integration test example. See [testing readers](development/testing.md) for numerical and linkage checks.

#### Testing multiple versions

When multiple versions of the raw data format are present, we encourage testing the reader on all of them to ensure backward compatibility. This task is greatly simplified if small test datasets are used for the CI tests. If this is not available, we suggest running the tests locally on multiple versions of the data before the PR is ready for review.

#### Testing that the visualization and table annotations are correct

Using `spatialdata-plot` and/or `napari-spatialdata` to visualize the data is a quick way to easily spot issues with spatial alignment or table annotations.

We encourage manually checking the visualization of the data. Please refer to the [documentation of `spatialdata-plot`](https://spatialdata.scverse.org/projects/plot/en/stable/api.html), to [its example notebooks](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks.html#technology-specific), and to the [documentation of `napari-spatialdata`](https://spatialdata.scverse.org/projects/napari/en/stable/notebooks/spatialdata.html). Please add such scripts to `spatialdata-sandbox` so that they are reproducible by others.

Two numerical checks make this visual review reproducible:

1. Check registration against a known landmark, spatial extent or mask pixel. For example, `test_registered_vertices_fall_on_their_mask_pixels` in `tests/integration/readers/xenium/test_reader.py` checks a transformed polygon against its own label.
2. Check annotations by matching a few known cell IDs and feature values to the source. The public Xenium tests in `tests/integration/readers/xenium/test_data.py` compare assigned transcripts with the gene counts. This catches cases where an image looks plausible but the wrong row supplies the colour.

#### Testing auxiliary functions

Exercise the parser or transformation at the module that owns it, then check the combined result in one reader workflow. For example, `tests/unit/readers/visium/test_layout.py` covers artifact discovery and `tests/integration/readers/visium/test_visium.py` checks the returned table, spots and images together. [Worked Visium example](development/visium-walkthrough.md)

### Adding a new converter

Readers are developed to parse raw data from spatial omics technologies. We also provide converters to convert data from/to other general data formats, i.e. data formats that are not specific to a single (or a small set) of spatial omics technologies.
An example of this is the legacy `AnnData` spatial format, used in early versions of `squidpy`.

Converters are not the primary scope of `spatialdata-io`, so we will just give some general indications: if you are interested in contributing a converter, please adapt and follow the guidelines as for readers (in particular, link to a specification, provide test data, and write extensive tests). Feedback is welcome: if you find it useful, we can expand this section of the contribution guide.

### Updating the CLI for readers and converters

Register a reader command under `src/spatialdata_io/_cli/commands/` and in the lazy CLI group. Keep the command's options consistent with the public Python reader, add its help and conversion tests, and check the generated reference. [CLI registration](development/readers.md#expose-document-and-verify-the-reader), [Command-line reference](cli.md)

## Bug tracking

When reporting a bug, include the reader call, a minimal input or public dataset
key, the producer version and the traceback. Run the following in the Python
session where you reproduced it and paste the output into the issue form.
`session-info2` is installed with spatialdata-io and reports the loaded
packages, their dependencies, Python version and operating system.

```python
import spatialdata_io
import session_info2

print(session_info2.session_info(dependencies=True))
```

To see current open [issues](https://github.com/scverse/spatialdata-io/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen) and [PRs](https://github.com/scverse/spatialdata-io/pulls?q=sort%3Aupdated-desc+is%3Apr+is%3Aopen) for a particular technology, please filter the issues/PRs by label. Not all the issues have been labeled yet, so additional issues/PRs for a particular technology may be present.

To provide an overview of the open bugs and to help prioritize them, we set up a [public (view-only) project board](https://github.com/orgs/scverse/projects/63/views/1). The project board can also list issues and PRs that are not in the `spatialdata-io` repository, for instance blockers in `spatialdata` or problems in downstream libraries that are linked to a bug in `spatialdata`/`spatialdata-io`. If you would like to update/edit the board, please reach out to us.

The issues in the project board need to be added manually, so the board may not be complete and we invite you to consult also the issues/PRs list. Due to the redundant nature of the project board, we are still experimenting with it, so please give us feedback if you find it useful or not.

## Supported technologies and specifications

See the [reader index](readers/index.md) for supported formats, experimental readers and links to version-specific layouts. For a new technology or version, include the producer's specification and representative files in the proposal. The [coordinate guide](user/coordinates.md) explains physical and pixel units, transformations and independent samples.

## Wrap-up: checklist

As a wrap-up, here is a checklist of the main points to consider when contributing a reader.

- Specification
    - [ ] Provide a link to the public specification/changelog of the raw data format (if no public specification available yet, reach out to scverse privately (Zulip/email))

- Reader implementation
    - [ ] Implement reader functions in `src/spatialdata_io/readers/`, or `src/spatialdata_io/experimental/` for experimental formats
    - [ ] Keep schemas and layout names in their reader-owned modules

- Example data
    - [ ] Provide (very) small public test dataset(s) (~100kB–10MB, preferred), licensed with a permissive license (e.g. CC BY 4.0)
    - [ ] Ensure dataset(s) cover edge cases of the format
    - [ ] In addition to the small dataset(s), consider creating scripts for downloading and converting a real dataset in `spatialdata-sandbox`:
        - [ ] `download.py` (fetch raw data → `data/`)
        - [ ] `to_zarr.py` (convert raw data → `data.zarr`)
        - [ ] Include `requirements.txt` (or add PEP 723 metadata to the scripts) for dependencies not included in `spatialdata`
    - [ ] If the data is not available online, upload it to the ["SpatialData Submissions" Zenodo community](https://zenodo.org/communities/scverse-spatialdata-submissions/)
    - [ ] We do not support private data, with the exception of data that will soon be made available to the public. In such cases, please reach out to us. Then, please provide a private repo with `download.py` and `to_zarr.py` (or upload the data privately to the "SpatialData Submissions" Zenodo community)

- Tests
    - [ ] Add test functions for the reader
    - [ ] Test multiple versions of the raw format if applicable
    - [ ] Verify visualization/alignment with `spatialdata-plot` or `napari-spatialdata`
    - [ ] Add proxy tests for spatial extent correctness (`spatialdata.get_extent()`)
    - [ ] Add proxy tests for coloring/annotations of selected elements/features
    - [ ] Add tests for auxiliary/helper functions if present

- Extra
    - [ ] Update the command and lazy registration under `src/spatialdata_io/_cli/` when the public reader or CLI changes
    - [ ] Consider contributing to the public project board (currently view only, please reach out for edits)
    - [ ] Provide feedback on this contribution guide to improve it further
