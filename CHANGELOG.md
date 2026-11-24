# Changelog

The format follows [Keep a Changelog][], and the project follows [Semantic Versioning][].

## Unreleased

### Added

- Generic GeoJSON imports can use top-level Feature IDs as shape indexes.
  [Generic GeoJSON](docs/readers/generic.md#geojson), [#249](https://github.com/scverse/spatialdata-io/issues/249)
- The reader guides show how to locate Xenium cells in H&E pixels and query
  Visium HD expression using QuPath polygons.
  [Xenium H&E coordinates](docs/readers/xenium.md), [Visium HD region queries](docs/readers/visium_hd.md#query-bins-with-a-qupath-annotation)
- Generic image readers support 2D and 3D grayscale and multichannel data, BigTIFF (`.btf`),
  explicit TIFF series and stored pyramid levels, and calibration from complete OME spatial
  sizes. [Generic images and GeoJSON](docs/readers/generic.md)
- Xenium can import secondary analysis on request. Run metadata, panel identities
  and selected protein feature annotations are available for supported bundles. When
  an H&E alignment matrix is missing, matching exported keypoints can supply the
  transformation. XOA 3.0 Prime layouts have also been verified with public bundles.
  [Xenium](docs/readers/xenium.md), [#180](https://github.com/scverse/spatialdata-io/issues/180)
- Stereo-seq reads selected native GEF counts without an analysis H5AD. The guide
  shows the supported SAW directory layouts and how to read flat exports.
  [Stereo-seq](docs/readers/stereoseq.md), [#161](https://github.com/scverse/spatialdata-io/issues/161), [#322](https://github.com/scverse/spatialdata-io/issues/322), [#355](https://github.com/scverse/spatialdata-io/issues/355)
- Visium HD selects raw or filtered cell counts independently of bin counts, can
  return bins without segmentation elements, and lets callers select the derived
  nucleus table independently of bin counts.
  [Visium HD](docs/readers/visium_hd.md), [#359](https://github.com/scverse/spatialdata-io/issues/359), [#390](https://github.com/scverse/spatialdata-io/issues/390)
- Readers that accept polygon geometry expose `invalid_geometries`. The default
  `"keep"` preserves source geometry. Use `"raise"` to reject invalid topology or
  `"drop"` to remove it according to the format guide.
- Dataset readers add the detected format version to the reader and package
  provenance in `attrs` when the source identifies it.
- The documentation now has a first-dataset tutorial using a public Xenium experiment,
  an offline image tutorial, task guides and a
  per-technology reference that identifies ISS as experimental. The table and
  coordinate guides explain physical XY, native pixels and grids, and region
  linkage. Contributors can follow a worked Visium reader through discovery,
  parsing, linkage, CLI registration and testing. The former guide URLs redirect.
  [First conversion](docs/user/first-conversion.md), [Reader guides](docs/readers/index.md),
  [Contributing](docs/contributing.md), [Developer walkthrough](docs/development/visium-walkthrough.md), [#340](https://github.com/scverse/spatialdata-io/issues/340)
- Contributors and CI can fetch optional test datasets by name or reader group
  using the same checksum-verified manifest. Unit and integration tests run
  offline when optional data are excluded. Contributors can register a public dataset,
  download it and run its integration test locally. [Test datasets](docs/development/datasets.md)

### Changed

- **Visium HD calibration.** Visium HD uses the manufactured 2 µm grid to place
  bins, segmented cells and registered images in slide micrometres. It reports
  the maximum and RMS registration error for the microscope image’s affine
  transform. Calibration is faster on position files stored in grid order.
  Shuffled rows remain supported, and duplicate sites are checked across the
  entire grid. [Visium HD coordinates](docs/readers/visium_hd.md#physical-coordinates)
- **Raster options.** Use `chunks` and `scale_factors` to control lazy rasters. The
  default is a single scale, one channel per image chunk, and 1024 × 1024 spatial tiles.
  Generic 3D images also use one Z plane per chunk. Pass `data_axes` as a string such as
  `"CYX"` where supported. TIFF reads follow native tiles or strips. PNG and JPEG decode
  an entire frame when computed. [Raster options](docs/user/rasters.md)
- **Standardized spatial calibration.** Readers with verified calibration express
  registered elements in micrometres, typically in `global`. Independent samples,
  cores and ROIs retain their own frames. For example, Xenium morphology pixels
  map into the same physical frame as cell centroids and transcripts. Classic
  Visium retains its full-resolution pixel frame. [Calibration](docs/user/coordinates.md),
  [#441](https://github.com/scverse/spatialdata-io/issues/441)
- **AnnData positions and counts.** Reader-produced `obsm["spatial"]` holds
  float64 `(x, y)` coordinates in micrometres. Where available, native positions
  are identified separately, for example in `obsm["spatial_pixels"]` or
  `obsm["spatial_grid"]`. For tables with declared positions,
  `uns["spatialdata_io_positions"]` records their axes, units, frame and meaning.
  Check position keys, linkage columns and identifier dtypes when combining new
  results with existing stores. Expression uses SciPy CSR arrays. [Tables](docs/user/tables.md)
- **CSV inputs.** Readers parse numeric fields using declared types and validate the full record width,
  including ignored columns. Integer fields such as the Visium spot grid reject decimal
  and exponent spellings. Undeclared annotations in CODEX CSVs and Curio cluster labels,
  metric values and defining features preserve literal text according to their schemas.
  [Tabular sources](docs/development/tables.md#tabular-sources-and-execution),
  [Curio Seeker](docs/readers/curio.md)
- **Dataset paths.** Discovered artifacts and relative overrides must resolve within the
  dataset root by default. Use `path_policy="relaxed"` or `--path-policy relaxed` for
  supported layouts with externally linked files, such as git-annex or DVC datasets.
  [Dataset paths and symlinks](docs/user/paths.md)
- **Generic.** Rename the Python `input` keyword to `path`. Pass options to
  `generic()` by keyword and supply `data_axes` for an image. GeoJSON rejects
  image-only options. Appending with `generic_to_zarr()` requires an explicit
  coordinate system. Use the CLI for write status output.
  [Generic images and GeoJSON](docs/readers/generic.md)
- **Visium and DBiT.** Visium uses `counts_source`, `positions_file`,
  `scale_factors_file` and `make_var_names_unique`. Query classic Visium in
  `<dataset_id>_pixels` and DBiT in `grid_pixels` or `image_pixels`.
  DBiT uses `table_file`, `barcode_positions_file`, `image_file` and
  `include_border`, and rebuilds table linkage to its barcode grid from source IDs.
  [Visium](docs/readers/visium.md), [DBiT](docs/readers/dbit.md)
- **CODEX, ISS and Steinbock.** Select CODEX tables and images with `cell_table_file`
  and `image_file`. The table suffix selects CSV or FCS. Use `image_to_table` for
  explicit registration. For ISS, use `from spatialdata_io.experimental import iss`
  and provide `raw_image_file`, `labels_file` and `table_file`. Use `masks_dir`
  to select Steinbock masks.
  Steinbock reads channel names from OME metadata and reserves `obs["cell_id"]` and `obs["region"]` for generated linkage.
  ISS and Steinbock each apply one `scale_factors` sequence to images and labels.
  [CODEX](docs/readers/codex.md), [ISS](docs/readers/iss.md), [Steinbock](docs/readers/steinbock.md)
- **seqFISH and MERSCOPE.** Use `include_*` options and seqFISH's
  `shape_geometry` to select products. In MERSCOPE, replace `vpt_outputs` with
  `vpt_directory`. The default reads all mosaic planes. Select a subset with
  `z_layers` or omit the mosaics with `include_images=False`.
  [seqFISH](docs/readers/seqfish.md), [MERSCOPE](docs/readers/merscope.md)
- **MCMICRO, CosMx and MACSima.** MCMICRO returns one linked table and keeps
  independent core and tissue-mask frames. It reads pipeline results and omits raw
  acquisition and illumination-correction images. CosMx preserves the native raster
  rows and expresses the vertical reflection in its transform. It uses `file_prefix` for file
  selection and returns one lazy transcript element. Its rasters and transcript
  points use `global_pixels` unless `pixel_size_um` supplies calibration. MACSima
  applies calibrated transforms automatically. Use `exclude_folder_names` and
  `exclude_cycles` to skip inputs.
  [MCMICRO](docs/readers/mcmicro.md), [CosMx](docs/readers/cosmx.md),
  [MACSima](docs/readers/macsima.md)
- **Stereo-seq and Visium HD.** Select Stereo-seq's products explicitly with
  `bin_sizes`, `cell_bins`, `analysis_files` and `image_roles`. Mask image
  roles now return labels. For Visium HD use `bin_sizes`, `bin_geometry` and
  `include_*` options to select outputs, which have fixed element names.
  [Stereo-seq](docs/readers/stereoseq.md), [Visium HD](docs/readers/visium_hd.md)
- **Xenium.** Use `include_*`, `feature_types`, `cell_geometry` and `table_target`
  to select products. The default table includes Gene Expression and available Protein
  Expression, and links to `cell_labels` by `label_id`. Polygon and circle targets
  link by `cell_id`. `global` is now in micrometres. [#198](https://github.com/scverse/spatialdata-io/issues/198)
  Label outputs require vendor
  masks. For stripped deposits, disable labels and select an available shapes target or
  an unlinked table. Decompress gzip-only transcript CSVs before reading them.
  `xenium_aligned_image()` requires `pixel_size`. `xenium_explorer_selection()`
  returns polygons indexed by selection with coordinates in micrometres.
  [Xenium](docs/readers/xenium.md)
- **Legacy AnnData.** Import both converters from `spatialdata_io.converters.legacy_anndata`.
  `to_legacy_anndata()` keeps table positions by default. Choose
  `position_source="centroids"` to derive positions from linked elements in one named
  coordinate frame, including when several regions share the frame. [#318](https://github.com/scverse/spatialdata-io/issues/318)
  `from_legacy_anndata()` creates linked points from legacy
  image-pixel coordinates and requires explicit association for multiple libraries.
  [AnnData access and legacy conversion](docs/user/legacy_anndata.md)
- The CLI mirrors reader options in kebab case, including repeatable pyramid
  factors. Run `python -m spatialdata_io <reader> --help` for the available
  options. [Command line](docs/cli.md)
- The package requires AnnData 0.13+, pandas 3+, PyArrow 21+, SciPy 1.15+,
  Zarr 3.2+, tifffile 2026.5.2+ and imagecodecs 2026.5.10+. Xenium conversion
  and writing to SpatialData Zarr are tested on pandas 3. Development checks
  use Ruff and Pyrefly. [Contributing](docs/contributing.md), [#364](https://github.com/scverse/spatialdata-io/issues/364)

### Fixed

- Classic Visium accepts MEX directories with any combination of plain and
  gzipped matrix, barcode and feature files. Duplicate spellings of the same
  member raise an ambiguity error. [Visium](docs/readers/visium.md), [#408](https://github.com/scverse/spatialdata-io/issues/408)
- DBiT reads a grayscale tissue image as one channel and preserves its pixels.
  [#267](https://github.com/scverse/spatialdata-io/issues/267)
- Visium HD derives nucleus counts from the raw 2 µm matrix and the vendor
  nucleus assignment, retaining contributions from assigned bins outside the
  tissue mask for the selected cells. [Nucleus counts](docs/readers/visium_hd.md)
- Classic Visium keeps the Space Ranger scale-factor JSON in
  `table.uns["spatial"][dataset_id]["scalefactors"]`. Visium HD records the
  nominal microscopy pixel size in `sdata.attrs["visium_hd"]["microns_per_pixel"]`
  and exposes a calibrated transformation on each registered image.
  [#223](https://github.com/scverse/spatialdata-io/issues/223), [#286](https://github.com/scverse/spatialdata-io/issues/286)
- Curio aligns Moran statistics to features by ID and retains them in `var`.
  CODEX resolves the selected processor CSV or FCS relative to the dataset
  directory. [Curio](docs/readers/curio.md), [CODEX](docs/readers/codex.md)
- Visium HD provides a named `global` frame even for files without a sample
  prefix. Use `include_images=False` for image-free output. CytAssist-only runs
  can omit the full-resolution microscopy image and its microscope transforms.
  [#422](https://github.com/scverse/spatialdata-io/issues/422), [#140](https://github.com/scverse/spatialdata-io/issues/140), [#270](https://github.com/scverse/spatialdata-io/issues/270)
- CosMx links cell IDs within each FOV and preserves case-variant annotations
  such as `cell_id` alongside the reserved `cell_ID`. A one-cell FOV can use
  the axis directions established by the other FOVs. [#291](https://github.com/scverse/spatialdata-io/issues/291), [#393](https://github.com/scverse/spatialdata-io/issues/393), [#213](https://github.com/scverse/spatialdata-io/issues/213)
- CosMx reads composite TIFF channels from the declared axes in supported FOV
  layouts, including channel-first stacks. [#403](https://github.com/scverse/spatialdata-io/issues/403)
- MERSCOPE projects valid cell boundaries across all Z slices, including cells
  absent from z0. It supports VPT Cellpose, Cellpose2 and Watershed micron-space boundaries.
  [#256](https://github.com/scverse/spatialdata-io/issues/256), [#208](https://github.com/scverse/spatialdata-io/issues/208)
- MACSima places planes at the integral pixel offsets declared by the source.
  Fractional positions or unsupported units raise a format error. [#419](https://github.com/scverse/spatialdata-io/issues/419)
- Xenium can read cell polygons and counts from supported single-polygon-set
  segmentation imports when unavailable labels and nucleus boundaries are
  disabled. For multinucleate cells in polygon-set bundles, the reader checks
  nucleus ownership against native cell label IDs. [#308](https://github.com/scverse/spatialdata-io/issues/308), [#402](https://github.com/scverse/spatialdata-io/issues/402), [#188](https://github.com/scverse/spatialdata-io/issues/188)
- Xenium preserves the declared morphology channels and reads aligned 2D
  images with arbitrary channel counts. [#226](https://github.com/scverse/spatialdata-io/issues/226), [#311](https://github.com/scverse/spatialdata-io/issues/311)
- Xenium uses the format version and image declarations in `experiment.xenium`
  to locate morphology. This also handles versioned bundles whose analysis
  software string is `R&D`. [#361](https://github.com/scverse/spatialdata-io/issues/361), [#362](https://github.com/scverse/spatialdata-io/issues/362)
- Xenium derives transcript feature categories from the count vocabulary.
  Row labels are unique across CSV partitions and Parquet row groups. The
  vendor identifier remains available in `transcript_id`.
  [#306](https://github.com/scverse/spatialdata-io/issues/306), [#397](https://github.com/scverse/spatialdata-io/issues/397)
- Visium, Visium HD and Xenium read 10x HDF5 counts directly. Visium and Xenium
  also read their Matrix Market inputs directly. The readers check integer precision
  before conversion to sparse floating-point arrays. Visium rejects competing modern
  and legacy feature files for the same matrix.
  [Raw-count precision](docs/user/tables.md#raw-count-precision)
- Curio, DBiT, ISS and Steinbock combine duplicate sparse entries in imported expression
  without integer overflow. If a sum exceeds the source integer dtype, the output
  widens to an exact floating type where supported and otherwise raises an error.
  [Expression and counts](docs/user/tables.md#expression-and-counts)
- seqFISH preserves unsigned 64-bit transcript cell IDs across CSV partitions, with
  zero reserved for unassigned transcripts. Polygon-only reads can use DAPI calibration
  without building an image graph. The seqFISH reader matches polygon Feature IDs
  to table cell IDs even if GeoJSON features are reordered.
  [seqFISH](docs/readers/seqfish.md), [#249](https://github.com/scverse/spatialdata-io/issues/249)
- Xenium accepts a decompressed `transcripts.csv` beside the original archive and
  preserves selected HDF5 feature annotations including genome and additional tags.
  [Xenium](docs/readers/xenium.md)

### Removed

- Generic decoder and model keyword mappings, Visium's arbitrary count keyword
  arguments, the generic image `use_tiff_memmap` switch, and MERSCOPE's `backend`
  selector. Use the explicit raster options listed in the reader guides.
- Xenium's public cell-ID codec helpers. Use the decoded `cell_id` supplied with
  counts, cell annotations or boundaries. [Xenium](docs/readers/xenium.md)
- The `include_images` export option for `to_legacy_anndata()`. Keep registered
  images and transformations in the SpatialData store.
  [AnnData access and legacy conversion](docs/user/legacy_anndata.md)
- Direct dependencies on `dask-image`, `joblib`, `scanpy`, `scikit-image`
  and `xmltodict`.

## v0.7.2

- Added pre-1.3 and CSV-only Xenium ingestion, contributed by @timtreis.
  This release can read gene counts from a 10x Matrix Market directory
  (`cell_feature_matrix/`) or its `cell_feature_matrix.tar.gz` archive when
  the HDF5 matrix is absent. [#427][xenium-427], [#287](https://github.com/scverse/spatialdata-io/issues/287)
- Adopted the scverse project template, contributed by @Zethson. The package began shipping
  `py.typed` and raised the minimum Python version to 3.12. [#411][template-411]

See the [v0.7.2 release notes][] for the full list of changes and credits. Release
notes for v0.7.1 and earlier are on the [Releases][] page.

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/
[releases]: https://github.com/scverse/spatialdata-io/releases
[v0.7.2 release notes]: https://github.com/scverse/spatialdata-io/releases/tag/v0.7.2
[xenium-427]: https://github.com/scverse/spatialdata-io/pull/427
[template-411]: https://github.com/scverse/spatialdata-io/pull/411
