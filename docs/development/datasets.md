# Download and register test datasets

The repository uses small synthetic inputs for offline tests and public datasets
to check readers against real instrument output. For local development, download
the dataset that reproduces the format version or behavior you are changing.
The same registry supplies the data preparation job in CI.

## Download a dataset and run its tests

From the repository root, list registered keys, groups, installation directories
and sources. The following commands install the tiny Xenium Prime mouse ileum
bundle and check its morphology pixels against the vendor TIFF.

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv sync --group dev --group test
uv run --no-sync python scripts/test_data_downloader --list
uv run --no-sync python scripts/test_data_downloader --dataset xenium_prime_mouse_ileum
uv run --no-sync pytest tests/integration/readers/xenium/test_data.py -k 'morphology_focus_reproduces and xenium_prime_mouse_ileum'
```

::::
::::{tab-item} Hatch
:sync: hatch

```bash
hatch run python scripts/test_data_downloader --list
hatch run python scripts/test_data_downloader --dataset xenium_prime_mouse_ileum
hatch test -- tests/integration/readers/xenium/test_data.py -k 'morphology_focus_reproduces and xenium_prime_mouse_ileum'
```

::::
:::::

| Task | Downloader options |
| --- | --- |
| Fetch one dataset | `--dataset xenium_prime_mouse_ileum` |
| Fetch a reader's registered datasets | `--group xenium` |
| Combine selections | Repeat either option. The downloader takes their union in registry order. Without selectors it downloads **all** datasets. |
| Share downloads between checkouts | Pass `--output /path/to/test-data` and set `SPATIALDATA_IO_TEST_DATA_DIR=/path/to/test-data` for pytest. The default for both is `data/` from the repository root. |
| Refresh an existing dataset | Add `--force` to the selected download. By default an existing destination is skipped without re-verification. |

Pooch verifies the download before installation. The downloader stages each
dataset under the output parent. A failed fetch or extraction leaves an
existing dataset in place. If several datasets are selected, the remaining
downloads are attempted and the command reports failed keys with exit status 1.

## Register a dataset

1. **Choose the evidence.** Identify the producer, format or software version,
   and behavior the example exercises. Prefer the smallest representative public
   release with a stable URL or version-specific DOI. Include the original
   dataset or licence page in a manifest comment and record the actual licence
   or permission in `source`. CC BY 4.0 permits redistribution and adaptation
   with attribution. Public download access alone does not establish those
   permissions. The contribution guide explains the project's data-sharing
   requirements and how to submit examples. [Example data and licensing](../contributing.md#the-importance-of-public-data-and-permissive-licenses)
2. **Register the layout.** Add a `[[datasets]]` entry to
   `scripts/test_data_downloader/datasets.toml` alongside the other entries
   for the reader. A test refers to the stable `key`. The CLI also accepts
   the reader-oriented `group`. Both start with a lowercase letter or digit
   and may contain lowercase letters, digits, underscores and hyphens.
   Choose a unique key and a unique, single-component `extracted_dir`.
   Set `test_path` if the reader's input directory is nested inside it.
   Paths are relative and cannot contain `..`.
3. **Choose one source form** from the table below. For each direct URL,
   compute SHA-256 over the exact upstream bytes, before extracting or
   renaming anything. Compare with the publisher's checksum where available.
   Store the digest as `sha256:` followed by 64 hexadecimal characters.
4. **Check the installation and the reader.** Validate the whole manifest
   with `--list`, fetch the new key into a fresh directory, inspect the
   resulting layout, and run its integration test against that directory.
   Commit the manifest and tests. Downloaded files stay outside Git.

| Source form | Manifest fields and installation | Existing example |
| --- | --- | --- |
| One ZIP | Set `url` and `known_hash` on the dataset. Extraction keeps the ZIP's directory hierarchy. | `seqfish` sets `test_path = "instrument 2 official"` for its nested reader input. |
| DOI repository | Set `doi` to the bare version DOI. Pooch fetches every file and verifies it using the repository's published registry. | `macsima_omap10` uses `10.5281/zenodo.18196366`. |
| Selected files | Add one `[[datasets.assets]]` per URL with its own `known_hash` and unique relative `target`. By default this names a file. With `extract = true` it names the extraction directory for a tar archive. Use `target = "."` to extract at the dataset root. | `xenium_mouse_brain_subset` keeps selected outs members. `visium_v2_1_mouse_brain` combines individual downloads with an extracted tar archive. |

For example, the registration for the small Xenium ZIP is complete with these fields:

```toml
[[datasets]]
key = "xenium_prime_mouse_ileum"
group = "xenium"
extracted_dir = "Xenium_Prime_Mouse_Ileum_tiny_outs"
source = "10x Genomics Xenium 3.0.0, CC BY 4.0"
url = "https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Mouse_Ileum_tiny/Xenium_Prime_Mouse_Ileum_tiny_outs.zip"
known_hash = "sha256:72dc2353825f6049959fa85e3c0617abd2a7cfb1c19531a2bba2d246fa276cd3"
```

For an asset update, verify the new URL and digest against the intended release
and repeat the fresh install. The command below prints a manifest-ready checksum
for a downloaded file, then exercises the manifest and reader. Substitute the
local filename, key and test selection for your source.

:::::{tab-set}
::::{tab-item} uv
:sync: uv

```bash
uv run --no-sync python -c 'import pooch, sys; print("sha256:" + pooch.file_hash(sys.argv[1]))' source.zip
data_dir=$(mktemp -d)
uv run --no-sync python scripts/test_data_downloader --list
uv run --no-sync python scripts/test_data_downloader --dataset xenium_prime_mouse_ileum --output "$data_dir"
uv run --no-sync pytest tests/unit/test_download_test_data.py
SPATIALDATA_IO_TEST_DATA_DIR="$data_dir" uv run --no-sync pytest -m 'xenium and data' -k xenium_prime_mouse_ileum
```

::::
::::{tab-item} Hatch
:sync: hatch

```bash
hatch run python -c 'import pooch, sys; print("sha256:" + pooch.file_hash(sys.argv[1]))' source.zip
data_dir=$(mktemp -d)
hatch run python scripts/test_data_downloader --list
hatch run python scripts/test_data_downloader --dataset xenium_prime_mouse_ileum --output "$data_dir"
hatch test -- tests/unit/test_download_test_data.py
SPATIALDATA_IO_TEST_DATA_DIR="$data_dir" hatch test -- -m 'xenium and data' -k xenium_prime_mouse_ileum
```

::::
:::::

## Use the fixture in a test

Put the test under `tests/integration/readers/<reader>/`. The shared
`require_test_dataset` fixture resolves `extracted_dir/test_path` below the
test data parent, skips when that directory is absent and gives the test the
`data` marker through the collection hook. The test directory supplies the
reader marker. For example, the public seqFISH dataset has 62 cells and 12
features in its first ROI:

```python
from collections.abc import Callable
from pathlib import Path

from spatialdata_io import seqfish


def test_seqfish_roi_dimensions(require_test_dataset: Callable[[str], Path]) -> None:
    sdata = seqfish(require_test_dataset("seqfish"))
    assert sdata.tables["table_Roi1"].shape == (62, 12)
```

Compare scientific results with independently read source values where possible.
Use pytest's `tmp_path` for preparation or Zarr writes so tests can share a
read-only download. Keep an offline synthetic fixture for each supported layout.
The manifest records source terms for each dataset, and reader integration
tests specify which products and versions each fixture exercises. [Testing workflows](testing.md)
