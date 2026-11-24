# Add or update a reader

## Establish the format contract

1. Link the vendor specification, release notes or another public description of
   the layout in the issue or pull request. Record producer versions, required
   and optional artifacts, supported encodings and the evidence for coordinates
   and units.
2. For an existing reader, capture the current output with deterministic
   fixtures before refactoring. Include each affected layout and mode, and
   check axes, dtypes, values, order, identifiers, transforms, table linkage,
   optional products and failure behavior. Document and test intentional
   scientific corrections.
3. Use a small synthetic fixture for each supported layout or version. Add
   independently verified real data where it can establish behavior the
   synthetic fixture cannot. [Testing workflows](testing.md)

## Use the common reader parameters

Every dataset reader takes `path` (a local `str` or `Path`) as its first argument
and returns a `SpatialData` object. Put optional selections in keyword
arguments. For example, `visium("outs", fullres_image_file="slide.tif")` resolves
the image relative to `outs`, regardless of the process's working directory.
A scan reader can also accept an individual file. Define which adjacent
artifacts participate in that case, which directory anchors relative overrides,
and how an explicit scan choice interacts with directory discovery. The CLI
must accept the same input kinds (`scan_io_options` for file or directory,
`dataset_io_options` for a directory) and write to `--output`. Python callers
choose whether and where to call `sdata.write()`.

| Option | Where it applies | Convention |
| --- | --- | --- |
| `path_policy` | Dataset directories | Default to `"strict"`. Explicit absolute file overrides may be external. Discovered artifacts and relative overrides must resolve inside the root unless the caller selects `"relaxed"` for symlinked data. [Path rules](../user/paths.md) |
| `scale_factors`, `chunks` | Images and labels | Default to one scale and one channel per 1024 × 1024 spatial chunk. Labels use the spatial sizes. Expose factors and sizes as repeatable CLI options. [Raster options](../user/rasters.md) |
| `invalid_geometries` | Readers with polygonal shapes | Default to `"keep"`. Define which linked rows or elements `"drop"` removes. `"raise"` identifies invalid geometries. |
| `include_*` or product selections | Optional format products | Use booleans for independent switches and explicit names for format choices, such as `bin_sizes` or `z_layers`. Discover only the artifacts needed for the selected products and their dependencies. |
| `data_axes`, `tiff_series`, `tiff_level` | Formats that permit an explicit raster selection | Accept the applicable options directly. Describe source axes as a string and TIFF selections as zero-based indices. |

Use a format-specific identifier only where the source or output needs one,
such as Stereo-seq's `sample_id`. Keep generic single-file reads separate:
`generic(path, ...)` returns **one element** and has no dataset-root
`path_policy`. [Generic files](../readers/generic.md)

## Implement one path from layout to result

Keep the public function short. Validate options, discover the layout once,
read bounded metadata, build requested elements, link completed tables,
assemble `SpatialData` and attach provenance.

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Export the public reader and any intentional public operations. |
| `_reader.py` | Validate public options and orchestrate products, dependencies and assembly. |
| `_layout.py` | Normalize the root, detect the format version and resolve artifacts. Pair version-specific sources with their encoding and column types. |
| `_schemas.py` | Declare tabular types and roles, required and optional fields, nullability, duplicate and additional-column policy, and reader-owned column validation. |
| `_tables.py` | Open table sources, project fields, build expression, align annotations and prepare the owned AnnData. |

Use a frozen, slotted layout with validated paths. Give mutually exclusive
versions or modes separate records when their required artifacts differ. Keep
parsing and model construction in product modules that accept the validated
paths and the facts they need. Introduce modules such as `_images.py`,
`_labels.py`, `_transcripts.py` and `_geometry.py` for products with distinct
contracts. Omit modules for products the format does not supply.

Discovery resolves ambiguity before expensive reads and gives all products the
same view of the dataset. A schema test can then validate a column role without
loading a raster or constructing a SpatialData object. Validate cross-artifact
relationships where the inputs meet. For example, a schema can validate barcode
spelling, while table assembly can check that every expression barcode has a
position. Linking the completed table after filtering and target construction
keeps each declared region and identifier aligned with the result.

Use the [implementation contracts](contracts.md#choose-a-shared-utility) to
choose the right decoder and validation boundary. Keep coordinate meaning,
feature roles, identity and geometry filtering in the reader. Generic image
and GeoJSON entry points build a single element directly and need no dataset
layout object.

For a complete example with real filenames and expected output, see [How the Visium reader works](visium-walkthrough.md).

## Write module docstrings

Every new or substantially changed reader implementation module, including
private modules and shared reader utilities, must have a NumPy-style module
docstring. Start with a one-line summary. Follow it with enough detail for a
maintainer to trace the input through the decisions the module owns to its
output. A package initializer that only re-exports an entry point can name
that entry point and its scope.

Document the facts a maintainer would need to preserve when changing the code.
The relevant details depend on the module's responsibility.

- **Layout and schema.** Name the actual artifacts, fields or version markers.
  Explain how the reader identifies alternatives and handles ambiguity. Use a
  short file tree or an example record when the distinction depends on names.
- **Scientific meaning.** State how source axes, units, identifiers and order
  map to the result. Explain the evidence for a coordinate transform or table
  linkage. For example, QPI channels follow physical TIFF page order after
  filtering on the XML image type. A thumbnail can match a measurement's
  dimensions.
- **Execution and ownership.** Identify what is read eagerly, what is deferred,
  which selections bound a read, and who owns any resource needed later.
  For example, the TIFF task reopens its file and can read several contiguous
  one-row strips in one operation.

The public reader function documents the complete caller-facing contract.
Keep each module docstring focused on that module's decisions. When those
decisions change, update the docstring in the same change and verify its
claims against the implementation and the fixtures for the affected formats.

## Expose, document and verify the reader

- **Stable Python API.** Re-export the function and list it in `__all__`
  in the reader package. Add the public name and module path to
  `_LAZY_IMPORTS` in `src/spatialdata_io/__init__.py`, add its name to
  the root `__all__`, and add the typed import in the `TYPE_CHECKING`
  block. Preserve the public import path when restructuring a reader.
- **Experimental Python API.** Place the reader under
  `src/spatialdata_io/experimental/` and export it from that namespace.
  Follow the example of `spatialdata_io.experimental.iss`. Document and
  test its explicit artifact layout.
- **CLI.** Add the Click command under
  `src/spatialdata_io/_cli/commands/`. Import the command in
  `src/spatialdata_io/_cli/__init__.py` and register it with
  `cli.add_command()`. Keep the scientific reader import inside the command
  so CLI help can load independently. Reuse the shared input/output, path
  and raster options from `_cli/_common.py` where applicable. Match the
  reader's option names, types, defaults, cardinality and selections, and
  pass the result to `write()` for dataset-to-Zarr commands. Check options
  with `python -m spatialdata_io` followed by the command name and `--help`.
- **Documentation.** Add a NumPy-style docstring and list the function in the
  appropriate autosummary block in `docs/api.md`. Add its guide to the
  appropriate toctree in `docs/readers/index.md`. Describe the expected file
  trees, reader options, products, coordinate and linkage semantics, deferred
  work and failure behavior. Sphinx generates the CLI reference from the
  registered command, so keep Click help and the guide's CLI examples
  consistent with the reader.
- Declare new direct runtime imports in `pyproject.toml`. Check source, tests,
  CLI, docs and packaging for usages before removing a dependency.
- Run the [checks for a reader change](testing.md#verify-a-reader-change)
  and [build the documentation](../contributing.md#building-the-docs-locally).

## Add a converter

Converters live under `src/spatialdata_io/converters/`. Define the source and
target models, mutation and ownership rules, and how the conversion treats
axes, coordinate frames, attachments and table linkage. Use the same tests
for numeric values, identity and persistence as appropriate. Document any
information the target representation cannot carry.
