# Contributing guide

This document aims at summarizing the most important information for getting you started on contributing to this project.
We assume that you are already familiar with git and with making pull requests on GitHub.

For more extensive tutorials, that also cover the absolute basics,
please refer to other resources such as the [pyopensci tutorials][],
the [scientific Python tutorials][], or the [scanpy developer guide][].

[pyopensci tutorials]: https://www.pyopensci.org/learn.html
[scientific Python tutorials]: https://learn.scientific-python.org/development/tutorials/
[scanpy developer guide]: https://scanpy.scverse.org/page/dev/

(contributing-readers-and-converters)=

For a reader or converter contribution, see the [reader submission guide](contributing_io.md), including [public data and permissive licenses](contributing_io.md#the-importance-of-public-data-and-permissive-licenses) and the [submission checklist](contributing_io.md#wrap-up-checklist).

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
