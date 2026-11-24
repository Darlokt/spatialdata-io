# Reader benchmarks

These measurements compare upstream main at `51ef276f` and the rewritten
readers at `aa444f95` on the same machine and input files. Peak process memory
fell across all 16 workflows. Time depended on the operation. DBiT read the
mouse embryo in 0.204 s versus 0.357 s. The tiled TIFF crop became faster, while
decoding the complete TIFF and writing the small Xenium Prime dataset took longer.

## Published datasets

Dimensions for tables are observations × features. Times include reader
construction. Rows marked **write** also complete a local SpatialData Zarr
write. The MACSima row computes the complete returned image. Other raster
products remain lazy until accessed. Each column reports the median of three
fresh processes per revision.

| Workflow | Main (s) | Branch (s) | Main (MiB) | Branch (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Visium mouse brain, 2,702 × 32,285 | 0.489 | 0.410 | 766 | 560 |
| Visium mouse brain, read and write Zarr | 1.126 | 1.072 | 766 | 592 |
| Visium HD, 8 µm bins and cell segmentation | 0.486 | 2.506 | 900 | 567 |
| DBiT E10 embryo, 936 × 22,802 | 0.357 | 0.204 | 656 | 491 |
| MACSima OMAP10, read and compute 4 × 77 × 94 pixels | 0.141 | 0.125 | 413 | 287 |
| seqFISH, 62 × 12 table only | 0.010 | 0.005 | 379 | 278 |
| Xenium Prime ileum, 23 × 5,006 table only | 0.040 | 0.056 | 403 | 308 |
| Xenium breast, 7,275 × 280 table only | 0.044 | 0.042 | 408 | 303 |
| Xenium Prime ileum, read and write Zarr | 1.005 | 1.331 | 542 | 371 |

Visium HD returned 20,830 filtered 8 µm bins and 612 segmented cells, both
with 32,285 features, plus their shapes and two display images. Construction
took about 5.2 times as long on the branch and used 333 MiB less peak memory.
In a separate profiled branch run, fitting and verifying the slide frame
accounted for 2.20 s of 2.61 s. The reader scans all 11,222,500 positions on
the 2 µm grid in bounded batches to establish the calibrated frame. This cost
also applies when only a coarser bin size is requested in this workflow.
[Visium HD coordinates](../readers/visium_hd.md)

## Controlled fixtures

| Workflow | Main (s) | Branch (s) | Main (MiB) | Branch (MiB) |
| --- | ---: | ---: | ---: | ---: |
| TIFF, one 256 × 256 crop in one channel | 0.116 | 0.078 | 309 | 274 |
| TIFF, all 4 × 2,048 × 2,048 pixels | 0.197 | 0.353 | 367 | 341 |
| MERSCOPE, 8,192 × 128 table only | 0.037 | 0.023 | 407 | 305 |
| CODEX, 30,000 × 2 CSV | 0.037 | 0.036 | 397 | 311 |
| CosMx, 4 × 4 table and FOV rasters | 0.033 | 0.114 | 381 | 294 |
| MCMICRO, 2 × 2 table, registered WSI and mask | 0.016 | 0.028 | 379 | 281 |
| Steinbock, 4 × 2 table and region rasters | 0.028 | 0.045 | 381 | 279 |

Both generic TIFF cases use the same seeded uint16 Deflate image with four
channels, 256 × 256 native tiles and requested 1 × 256 × 256 chunks. The
crop selects rows and columns 768–1023 of channel zero. Distinguishing that
read from a full decode matters for workflows that display or analyze a
small region. The four-cell CosMx and Steinbock cases mainly measure setup;
larger experiments can have different costs.

The [startup and matched-workload analysis](performance-diagnostics.md) measures
fresh imports and repeated calls independently, and explains where the slower
reads and writes spend their time.

## What was verified

- For every timed table, we compared the full canonical CSR values and indices,
  dimensions, and observation and feature order. All numerical matrices matched.
  CosMx changed observation names for the same rows, and MCMICRO and Visium HD
  changed table keys. The MCMICRO fixture contains only the common registered image,
  segmentation and quantification products. Steinbock preserves raw image and
  label values and now assigns OME channel names. CosMx preserves source raster
  row order where main flipped the Y axis. Counts match for the same cells. The format guides describe changes to frames, dtypes and linkage.
- TIFF crops and full images matched in shape, dtype and every pixel. The MACSima
  stacks also matched in channel order. Separate checks reopened the Visium and
  Xenium Zarr stores and compared expression, ordered row and feature names,
  image and label pixels, shape coordinates, and available transcript XYZ rows
  across revisions.
- Three additional comparisons did not produce equivalent workflows. On the
  shared AnnData version, upstream ISS calls the removed `anndata.read`. Upstream
  Curio rejects the fixture's nonempty Moran statistics when comparing their row
  index with feature IDs. Both fixtures read successfully on the branch.
  Upstream Stereo-seq returns analysis and geometry products alongside bins on
  the compact GEF fixture. The branch bin-only selection yields a different
  product set. For the common bin table, counts matched by gene and observation.
  The branch retains source gene order. These three cases have no reported timing ratio.

## Reproduce the comparison

Run from the branch checkout on Linux. The runner alternates revision order,
starts a new worker for every measurement, and imports each implementation
from its checkout. It uses the synchronous Dask scheduler, sets single-thread
limits for BLAS and Arrow, and measures wall time after importing the reader.
Peak RSS is the process high-water mark through the end of the workflow,
including imports. Validation runs afterward. Reads used a warm filesystem
cache, including source data for the first timed trial. Zarr writes used a
fresh directory under the system temporary directory on tmpfs.

Measured on an Intel Core i7-13700K under Linux 7.0.0, restricted to two logical
CPUs. Both checkouts used Python 3.14.4, SpatialData 0.8.0, AnnData 0.13.3.post0,
pandas 3.0.5, NumPy 2.5.2, SciPy 1.18.1, PyArrow 25.0.1, Dask 2026.8.0,
tifffile 2026.8.23 and Scanpy 1.12.4. These measurements describe local reads
and writes with those versions. Storage latency, scheduler and library versions
can change the results.

1. Create a second checkout of the baseline and install dependencies for both
   readers in one development environment. The example uses uv.

   ```bash
   git worktree add --detach ../spatialdata-io-main 51ef276f7508d24d6072b69bea777dede9952812
   uv sync --group dev --group test
   uv pip install --python .venv/bin/python scanpy dask-image joblib scikit-image xmltodict
   ```

2. Install the [test datasets](datasets.md) and prepare shared fixtures once.
   The DBiT preparation uses the published coordinate-to-barcode mapping.
   The remaining synthetic inputs use the repository's reader test fixtures.

   ```bash
   .venv/bin/python scripts/test_data_downloader \
     --dataset visium_v2_1_mouse_brain --dataset visium_hd_tiny --dataset dbit_mouse_e10 \
     --dataset macsima_omap10 --dataset seqfish \
     --dataset xenium_prime_mouse_ileum --dataset xenium_breast
   bench_dir=$(mktemp -d)
   .venv/bin/python -m scripts.benchmark_readers.prepare data "$bench_dir/fixtures"
   ```

3. Run the comparison. Select two available CPU IDs on your machine.
   The runner writes one JSON per workflow, revision and trial, including
   product hashes. Use `--cases generic-crop generic-full` to run a subset.

   ```bash
   taskset -c 2,3 .venv/bin/python -m scripts.benchmark_readers.run \
     ../spatialdata-io-main data "$bench_dir/fixtures" "$bench_dir/results"
   ```

{download}`Download all 96 individual measurements (CSV) <../_static/reader-benchmarks.csv>`.
The CSV includes construction time separately from total workflow time.
