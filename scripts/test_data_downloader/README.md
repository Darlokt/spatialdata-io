# Optional test-data downloader

This directory contains the manifest and command-line tool used to download the
external datasets required by some `spatialdata-io` integration tests. These
datasets are intentionally not stored in Git because they are large or are
maintained by another project.

The downloader provides one reproducible entry point for local development and
CI. It:

- reads dataset registrations from [`datasets.toml`](datasets.toml);
- verifies every downloaded file against a published or committed checksum;
- assembles multi-file datasets in a temporary staging directory;
- leaves an existing dataset unchanged if a download or extraction fails; and
- makes registered datasets available to tests through the shared
  `require_test_dataset` fixture.

The manifest is project metadata, not a copy of the Pooch registry. It records
stable test keys, reader groups, local directory names, provenance, and one of
the supported download-source configurations.

## Using the downloader

Run commands from the repository root with the development dependencies
installed (`uv sync --group dev --group test`).

List the registered keys, groups, output directories, and sources:

```bash
uv run python scripts/test_data_downloader --list
```

Download one dataset or all datasets in a reader group:

```bash
uv run python scripts/test_data_downloader --dataset visium_v1_human_lymph_node
uv run python scripts/test_data_downloader --group visium
```

`--dataset` and `--group` may each be repeated. When both are present, the
downloader installs the union of the selected keys and groups in manifest
order. With neither option, it downloads every registered dataset.

Datasets are installed below `data/` by default. Use another parent directory
with `--output` and point pytest at the same location:

```bash
uv run python scripts/test_data_downloader \
  --dataset visium_v1_human_lymph_node \
  --output /path/to/test-data
SPATIALDATA_IO_TEST_DATA_DIR=/path/to/test-data uv run pytest -m "visium and data"
```

An existing destination is skipped. Pass `--force` to download a fresh copy and
replace that destination after the new dataset has been completely staged:

```bash
uv run python scripts/test_data_downloader --dataset visium_v1_human_lymph_node --force
```

The downloader attempts every selected dataset and exits with status 1 after
reporting all failures. Downloads require network access; normal unit tests and
manifest validation do not.

## Manifest fields

Every `[[datasets]]` entry defines these fields:

| Field | Meaning |
| --- | --- |
| `key` | Stable lowercase identifier used by the CLI and pytest fixture. |
| `group` | Lowercase reader-oriented selector used by `--group`. |
| `extracted_dir` | Single directory name created below the output directory. |
| `source` | Human-readable upstream provenance, format version, and license. |
| `test_path` | Optional directory below `extracted_dir` returned to a test. |

Keys and groups may contain lowercase letters, digits, underscores, and
hyphens. Paths must be relative and may not contain `..`. Keep keys and output
directory names stable: tests, local caches, and CI workflows can refer to them.

Each dataset uses exactly one of the following source forms.

### One ZIP archive

Use a direct, stable URL and pin the complete ZIP by SHA-256. The archive's
contents are extracted into `extracted_dir` without removing a top-level
directory.

```toml
[[datasets]]
key = "example"
group = "example_reader"
extracted_dir = "example_outs"
source = "Example project 2.0, CC BY 4.0"
url = "https://example.org/releases/2.0/example.zip"
known_hash = "sha256:<64 hexadecimal characters>"
```

### DOI repository

Prefer a stable DOI when its repository publishes a Pooch-compatible file
registry. Pooch obtains the repository's file list and published hashes, then
downloads and verifies every registered file.

```toml
[[datasets]]
key = "example"
group = "example_reader"
extracted_dir = "example_outs"
source = "Example project 2.0, CC BY 4.0"
doi = "10.5281/zenodo.1234567"
```

Store the bare DOI, without a URL or `doi:` prefix. A DOI entry cannot select a
subset of repository files; use independently hashed assets if the test needs
only part of a large publication.

### Independently hashed assets

Use `[[datasets.assets]]` when one logical dataset is assembled from multiple
downloads or when retaining a small set of upstream artifacts avoids a much
larger archive. Every asset has its own URL, checksum, and destination.

```toml
[[datasets]]
key = "example"
group = "example_reader"
extracted_dir = "example_outs"
source = "Example project 2.0, CC BY 4.0"

[[datasets.assets]]
url = "https://example.org/releases/2.0/counts.h5"
known_hash = "sha256:<64 hexadecimal characters>"
target = "filtered_feature_bc_matrix.h5"

[[datasets.assets]]
url = "https://example.org/releases/2.0/spatial.tar.gz"
known_hash = "sha256:<64 hexadecimal characters>"
target = "."
extract = true
```

For a regular file, `target` includes the destination filename and parent
directories are created automatically. With `extract = true`, the asset must be
a tar archive and `target` is its extraction directory; `target = "."` extracts
at the dataset root. Tar members are filtered to reject path traversal and
unsafe filesystem objects. Asset targets must be unique within an entry.

## Adding or updating a dataset

1. Choose the smallest public, stable, and permissively licensed upstream data
   that faithfully covers the format behavior under test. Record the data
   producer, format/software version, and license in `source` and a nearby TOML
   comment with the authoritative dataset or license page.
2. Decide whether a DOI, one ZIP archive, or independently hashed assets best
   represents the source. Do not mix source forms in one dataset entry.
3. Download each direct URL once and calculate its SHA-256 digest. Pooch can
   produce the exact manifest value:

   ```bash
   uv run python -c 'import pooch; print("sha256:" + pooch.file_hash("/path/to/download", alg="sha256"))'
   ```

4. Append the entry to `datasets.toml`, keeping related reader datasets
   together. Never use an unknown hash, a mutable URL without a pinned digest,
   credentials, or a private machine path.
5. Validate the registration and perform a clean download into a disposable
   output directory:

   ```bash
   uv run python scripts/test_data_downloader --list
   test_output="$(mktemp -d)"
   uv run python scripts/test_data_downloader --dataset example --output "$test_output"
   ```

6. Inspect the resulting directory and connect the integration test to the
   stable key:

   ```python
   def test_example(require_test_dataset: Callable[[str], Path]) -> None:
       dataset_path = require_test_dataset("example")
       result = example_reader(dataset_path)
   ```

   The fixture skips cleanly when optional data is unavailable. Add the reader
   marker and let the fixture apply the repository's `data` marker.
7. Run the downloader unit tests and the affected data-backed integration test:

   ```bash
   uv run pytest tests/unit/test_download_test_data.py
   SPATIALDATA_IO_TEST_DATA_DIR="$test_output" uv run pytest -m "example_reader and data"
   ```

To add a file to an existing multi-asset dataset, append another
`[[datasets.assets]]` table with its own checksum and unused `target`, then
repeat the clean-download check. A ZIP or DOI entry cannot also contain assets;
either update that upstream source or deliberately convert the complete entry
to the multi-asset form. When changing an existing URL, file, archive contents,
or checksum, verify that the change is expected and rerun every integration test
that uses the dataset key.

Do not commit downloaded datasets, temporary output, credentials, or private
data. The repository tracks only the manifest and downloader implementation.

## DBiT-seq experimental mouse embryo dataset

```bash
uv run python scripts/test_data_downloader --dataset dbit_mouse_e10
uv run pytest tests/integration/readers/dbit/test_dataset.py
```

`dbit_mouse_e10` downloads about 97.0 MB: the
[E10 mouse embryo H5AD from versioned Zenodo record 12785822](https://zenodo.org/records/12785822/files/DBiT-seq_liu2020high_E10_whole_gene_best_data.h5ad)
from the experiment of [Liu et al. (2020)](https://doi.org/10.1016/j.cell.2020.10.026)
and the authors' [nucleotide-to-grid mapping](https://github.com/MingyuYang-Yale/DBiT-seq/blob/4487131b0ecfb1e490bf16829d7376ea5fbe1c9e/Datasets/barcode.xls)
at a pinned commit. The experimental source collection is distributed under
[CC BY 4.0 on Zenodo](https://zenodo.org/records/12785822). The nucleotide map
records the factual barcode sequences and grid coordinates from the authors'
repository. Both files are pinned by SHA-256 in the manifest.

The published H5AD contains 936 observations and 22,802 features. It has
coordinate observation names such as `41x14` and stores raw counts as a dense
`float32` array in `X`. It is therefore prepared explicitly inside the
integration test's temporary directory: the actual published map restores each nucleotide
observation identifier, while expression and feature order remain unchanged.
The published first coordinate corresponds to barcode B and the second to
barcode A. The reader returns its documented A/B ordering. The test retains
the source coordinate identifiers and coordinates as metadata and verifies every sparse expression
value, feature order, coordinate conversion, and linked shape membership.

The downloaded H5AD is not a native input layout accepted directly by
`dbit()`. Preparation preserves all features in this archived H5AD. It does not
claim to reproduce the full unfiltered experiment. The reader converts the source
dense expression to a canonical CSR sparse array. No histology image is
included, so this dataset validates the experimental count/barcode workflow.
Synthetic fixtures separately cover raster behavior. Downloaded and prepared
H5AD files remain outside Git.

The later [DBiTplus study](https://doi.org/10.1038/s41592-025-02948-0) also
deposits genuine DBiT spatial-transcriptomic data with paired imaging. For
example, [GEO sample GSM9239729](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9239729)
provides an FFPE mouse-embryo expression matrix, brightfield image, ROI mask and
spot-position list. Its processing metadata explicitly describes spatial
barcodes A and B and microfluidic-channel coordinates. These processed files
are not a supported `dbit()` input layout: the position file selects decoded
coordinate IDs rather than mapping nucleotide barcodes, and the authors'
image-grid spacing differs from this reader's square-grid convention. They
cannot serve as an image-alignment test without explicit handling of those
differences. They are from a different experiment and must not be paired with
the E10 table above. They are not downloaded by this fixture.

## Stereo-seq datasets

The `stereoseq` group prioritizes original company examples and adds licensed
study data for historical producer coverage:

| Key | Assets and provenance | Source data terms |
| --- | --- | --- |
| `stereoseq_mouse_eyeball` | Original C04687E314 unadjusted-cell, adjusted-cell and whole-chip raw square GEFs, plus registered H&E TIFF; vendor identifies SAW 8.1, patch unknown. Approximately 646 MB combined. | Public vendor downloads; no open redistribution or repacking grant has been established. Assets stay in the ignored local data directory. |
| `stereoseq_mouse_kidney_saw7` | Original C02533C1 cell and tissue GEFs; vendor identifies SAW 7.0, patch unknown. Approximately 1.28 GB combined. | Public vendor downloads; no open redistribution or repacking grant established. |
| `stereoseq_saw7_1_square` | Complete A03994E6 RNA tissue GEF from Zenodo 12595241; study identifies SAW 7.1.2. Approximately 1.06 GB. Protein counts are not selected. | CC BY 4.0. |
| `stereoseq_saw6_square` | Complete A01785D6 tissue bin1 GEF from Zenodo 12595241; study identifies ordinary transcriptomics as SAW 6. Approximately 192 MB. This covers the name-only feature schema and does not establish a SAW 7 run. | CC BY 4.0. |

Each original asset has a SHA-256 checksum in the manifest. Download and run the
reader workflows through the shared dataset fixture:

```bash
uv run python scripts/test_data_downloader --group stereoseq
uv run pytest tests/integration/readers/stereoseq/test_data.py -m 'data and slow'
```

The cell workflows compare every native count and exon against the complete
source for company SAW 7.0 unadjusted cells and SAW 8.1 unadjusted/adjusted cells.
They preserve semantic cell IDs, source feature order, coordinates, summaries
and target linkage, while sparse densification is forbidden. Registered-image
construction executes no pixel tasks; a bounded pixel selection is checked
independently. The vendor's nonsimple stored contours fail strict polygon mode.

The SAW 6 square workflow checks its complete source: more than twelve million
points and several GiB of memory. Larger SAW 7.0, 7.1.2 RNA and 8.1 raw square
sources use deterministic temporary reductions from the verified originals.
Each reduction contains complete leading feature spans, capped at 24 features
and 50,000 expression records, and retains native field dtypes, aligned exons,
origin, physical pitch and source extents. Values and first-occurrence coordinate
order are checked against the original file, independently of the reduced file.
This tests real source records and their storage profiles; it does not claim
whole-file reader execution for those larger square sources. No reduced vendor
artifact is committed or redistributed. All real-data tests are marked `data`
and `slow`; ordinary offline tests use synthetic files.

Stereo-CITE coverage concerns only the RNA partner. A separate offline test
reproduces the observed protein metadata's one-element `omics=Proteomics`
attribute and proves rejection before expression datasets are read. This is a
metadata-derived synthetic rejection test, not validation of the complete
3.2 GB protein file or support for paired RNA/protein output. When a historical
GEF omits `omics`, its physical fields alone cannot identify every modality;
select the known RNA file explicitly.

## Xenium datasets

The `xenium` group covers every bundle family with original 10x Genomics
downloads (CC BY 4.0). Sizes are the extracted local directories.

| Key | Assets and provenance | Covers |
| --- | --- | --- |
| `xenium_mouse_brain_subset` | Selected members of the FF mouse brain subset, XOA 1.0.2, with CSV and MatrixMarket twins. About 700 MB. | integer IDs, unversioned format, stripped deposits |
| `xenium_brain_addon` | Selected members of the FFPE human brain with add-on panel, XOA 1.3.0.5. About 970 MB. | first hexadecimal IDs, binary text columns |
| `xenium_heart` | Selected members of the FFPE human heart, XOA 1.9.0.1, with CSV/MatrixMarket twins, the aligned H&E image and its alignment matrix. About 1.1 GB. | hexadecimal IDs, aligned images |
| `xenium_breast`, `xenium_lung` | Complete two-FOV `outs.zip` bundles, XOA 2.0.0.10. About 370 and 270 MB. | first polygon sets, per-channel focus files |
| `xenium_prime_mouse_ileum`, `xenium_prime_multicell_mouse_ileum` | Complete tiny `outs.zip` bundles, XOA 3.0.0.15. About 18 MB each. | Xenium Prime, multimodal segmentation |
| `xenium_ovary`, `xenium_multicell_ovary` | Complete tiny `outs.zip` bundles, XOA 4.0.0.19. About 32 MB each. | `ch<NNNN>_<marker>` focus files |
| `xenium_protein_kidney` | Complete tiny `outs.zip` bundle, XOA 4.0.0.19. About 25 MB. | Protein Expression and the twelve-channel protein kit |

Member datasets are assembled from individually hashed files of the 10x
release (`<dataset>_<file>`) instead of multi-gigabyte archives. Download and
run the real-data workflows with:

```bash
uv run python scripts/test_data_downloader --group xenium
uv run pytest tests/integration/readers/xenium/test_data.py -m data
```

Every expected value comes from the raw files read independently of the
reader: counts through h5py, identifiers and polygons through Arrow, image
windows through tifffile, and the alignment matrix as text. The workflows check
exact counts and identifiers, the reconciliation of `qv >= 20` transcripts with
the count matrix, mask labels against `cell_area` and the registered
boundaries, OME channel names and pixel windows, protein scaling, the aligned
H&E frame, secondary analysis, and that reads assembled from the CSV and
MatrixMarket twins equal the full bundle.
