# Startup and matched reader workloads

The [main benchmark](performance.md) times work after resolving the public
reader function. For a script that reads a dataset once, importing the
scientific stack also matters. For a notebook processing a second dataset,
some imports and caches are already warm.

We measured both situations on identical inputs with the same product
selections. **First call** starts immediately before importing
`spatialdata_io` in a fresh Python interpreter and ends when the requested
read, pixel computation or write completes. **Repeat** times the second full
call in the same process, using a fresh output directory for writes. Each
value is the median of three independently started processes. Interpreter
launch, post-read checks and profiling are outside these times.

| Workflow | First call, main (s) | First call, branch (s) | Repeat, main (s) | Repeat, branch (s) |
| --- | ---: | ---: | ---: | ---: |
| TIFF crop | 0.891 | 0.838 | 0.027 | 0.007 |
| Full TIFF | 0.970 | 1.122 | 0.102 | 0.257 |
| CosMx, two FOVs | 1.104 | 0.897 | 0.025 | 0.022 |
| MCMICRO, registered WSI and mask | 1.065 | 0.823 | 0.011 | 0.021 |
| Steinbock, two samples | 1.072 | 0.830 | 0.022 | 0.030 |
| Visium HD, 8 µm bins and cells | 1.542 | 3.314 | 0.485 | 2.451 |
| Xenium Prime, table | 1.097 | 0.863 | 0.025 | 0.052 |
| Xenium breast, table | 1.110 | 0.842 | 0.032 | 0.028 |
| Xenium Prime, read and write Zarr | 2.078 | 2.149 | 0.870 | 1.194 |

## Where the time goes

- **CosMx.** The first branch read triggered an import of `distributed` during
  Dask rechunking in Xarray's raster model. That import occupied 0.133 s in
  a separate profiled 0.202 s read. Resolving the public reader itself took
  1.071 s on main and 0.789 s on the branch. The branch finished first both
  when imports were counted and when both implementations were warm. The
  benchmark disabled transcripts on both and requested the same two FOVs.
- **Generic TIFF.** A warm single-tile crop took 7 ms on the branch and 27 ms
  on main. Decoding every pixel took 257 ms and 102 ms respectively. The full
  branch read opened the TIFF for 256 bounded tile selections. The crop
  needed one. Main used four frame reads for the full image. Both
  implementations returned the same pixels and used the same requested output
  chunks. Reopening sources per tile costs time during a full serial scan.
- **Visium HD.** The two revisions read the same 8 µm expression and
  segmentation products. The branch still took 2.45 s warm. Two scans of the
  11.2 million native 2 µm positions fit and verify its physical slide frame.
  A separate profile measured 2.20 s for that step. Main returns its older
  coordinate contract. There is no public switch to skip calibration on the
  branch, so the timing includes that scientific difference.
- **Xenium.** For the 23-cell Prime table, warm time went from 25 ms on main
  to 52 ms on the branch. The branch also reads the optional 5K gene-panel
  metadata. With just `gene_panel.json` omitted from a copy of the same
  bundle on **both** revisions, the counts and identifiers still matched and
  the warm medians were 25 ms and 22 ms. The larger 7,275-cell breast table
  also favored the branch on repeat. The full Prime Zarr write remained
  slower. In the profiled branch write the two mask arrays made 256
  path-backed selections, each reopening the vendor ZIP. Turning both masks
  off on both revisions reduced warm read-plus-write time to 0.356 s on main
  and 0.559 s on the branch. Mask I/O explains part of the full-write gap.
- **Steinbock and MCMICRO.** Both branch readers finished sooner from a fresh
  import, and both took longer on repeat. For Steinbock, inspecting the four
  raster files, parsing OME metadata, and validating table IDs and linkage
  show up in the profile. The resulting channel names come from OME, and raw
  pixels matched main. For MCMICRO, we excluded the optional raw and
  illumination inputs, which only main returns. The shared registered image,
  mask and expression matched. In the separate branch profile, OME XML
  parsing took 10 ms and final table linkage took 5 ms. At this scale those
  fixed costs are a substantial share of a read.

## Reproduce the controls

After [preparing the benchmark inputs](performance.md#reproduce-the-comparison),
pass one of these options to the same runner and choose a new output directory
for each invocation:

| Option | Measurement |
| --- | --- |
| `--first-call` | Records public import time, call time and actual time from first import through completion in each worker. |
| `--warmup` | Completes the workload once, fingerprints its products, then times the next call. |
| `--warmup --cases xenium-no-panel xenium-no-masks-write` | Runs the two matched Xenium controls on both revisions. |
| `--profile --repeats 1` | Writes a separate cProfile per timed region. Inspect its `.pstats` file with `python -m pstats`. Use the unprofiled runs for elapsed-time comparisons. |

{download}`Download the 120 individual first-call, repeat and control measurements (CSV) <../_static/reader-benchmark-controls.csv>`.
