# Reader package template

A contributor should be able to find where a reader chooses its files, declares its columns, interprets coordinates and links a table before learning the vendor format. Use the same core filenames for dataset readers. Add product modules for the products the format actually supplies. Each file below owns a decision that can be tested independently.

## Package layout

```text
readers/<format>/
    __init__.py          public exports
    _reader.py           public transaction and product selection
    _layout.py           artifact discovery and typed source descriptions
    _schemas.py          source fields, normalized fields and output column contracts
    _tables.py           table I/O, alignment and construction
```

The 14 current dataset readers all produce tables and already have these five files. A future dataset without tabular input or output can omit the corresponding modules. Experimental readers use the same template under `experimental/`. Generic dispatches a **single file to one element**. Its image and GeoJSON functions own their input and output directly. It has no dataset discovery, expression table or final dataset assembly.

| File | What belongs here | What crosses its boundary |
| --- | --- | --- |
| `__init__.py` | Export the supported reader and any intentional public operations. Keep the package and CLI entry points aligned. | Public functions with concrete signatures. |
| `_reader.py` | Validate caller options, call `discover_layout()` once, request selected products, reconcile dependencies, assemble the result and attach provenance. Coordinate filtering and final linkage when they depend on several products. | Public inputs in, a `SpatialData` out. Private builders receive the relevant sources, selection or normalized facts. |
| `_layout.py` | Resolve the dataset root, overrides, versions, alternative layouts and required or optional artifacts. Associate each selected source with the encoding, schema and logical name that its consumers need. | Frozen, slotted records containing paths and small selection facts. Use alternatives for mutually exclusive modes and optional fields for independent products. Open resources end within discovery. |
| `_schemas.py` | Declare physical field types and order, field roles, required and optional columns, null and additional-column policy, identifiers, normalized and output column contracts. Validate headers, footers and decoded values where those facts first become available. | An Arrow schema, a typed selection of columns, validated identifiers or a normalized payload. Validators can take a path for diagnostics. File opening, scanning, joins and model construction belong to the consuming module. |
| `_tables.py` | Open the table artifact, apply the declared projection and types, build sparse expression, align annotations and assemble the owned AnnData. Apply final linkage here when all target facts are already available. | A prepared table or a final linked table. State which it is. Final region and instance columns refer to the filtered, ordered population. |

### Modules for additional responsibilities

| Add this file when needed | Responsibility |
| --- | --- |
| `_version.py`, `_metadata.py` | Isolate substantial version detection or parsing of experiment metadata. Return typed values with units. Discovery can ask for the metadata it needs to select a layout. If a version belongs to each channel file, parse it there and reconcile channels at stack planning. |
| `_images.py`, `_labels.py`, or `_rasters.py` | Use `_images.py` for images and `_labels.py` for an independently constructed mask product. Use `_rasters.py` when images and labels share one grid or loading contract. Select axes, channels, registration and model options here. The shared image foundation owns decoding, lazy tasks and file handles. |
| `_transcripts.py` | Project the transcript source and build lazy point partitions. Read the header or footer eagerly, apply the schema's row checks in the partition that reads the rows, and adapt coordinates and categories for the points model. |
| `_geometry.py` | Decode or construct shapes, align shape identities and apply the format's drop policy. The schema establishes required properties and identifiers. The geometry builder establishes topology, coordinate interpretation and consequences of dropping geometry. |
| `_positions.py` | Prepare positions when tables and geometry share coordinates. For example, Curio validates duplicate native XY arrays and writes calibrated positions to its owned table here. Its geometry builder takes only the native XY and bead identifiers. |
| `_transforms.py`, `_calibration.py` | Give a registration calculation or calibration reconciliation a home when multiple products need it or it has independent scientific tests. Name the source and destination frame, axes and units. A simple scale used by one product can stay with it. |
| Format-specific modules such as `_channels.py`, `_barcodes.py`, `_cells.py` or `_analysis.py` | Give a substantial storage format, algorithm or independent product one owner. For example, MACSima plans channel order and canvas placement once for both the image and its channel table. Keep the package's column declarations discoverable in `_schemas.py`. |

Keep imports flowing from the public entry point into discovery and product builders. Both discovery and builders may use the reader's schemas. Keep the schemas independent of the public entry point, the CLI and concrete model builders. Shared foundations can be imported by any reader and remain independent of vendor modules.

A product module can consume another product's result when there is a real dependency. A table can use the inspected mask dtype to check ID range. A transcript reader should be able to open a CSV header without importing an expression-table builder. Use an existing storage primitive directly. Extend a shared primitive only when independent consumers have the same semantics and can share its tests.

## From file to typed value

1. **Select the source.** Use the logical artifact name to choose its role and encoding. Keep that decision alongside its validated storage path. A resolved symlink can have a different basename. When filenames break ties in channel order, retain the order established by discovery across path policies. For format versions with different columns, pair the selected file with its version's field types during discovery.
2. **Declare the physical schema once.** For fixed CSV fields, use one ordered Arrow schema and derive names from it. For a variable panel, make a schema function that takes the validated header and evidence identifying expression features. For Parquet, inspect the footer before reading batches. Declare allowed physical representations and checked conversions. Keep native HDF5 compound types, FCS event buffers, TIFF metadata and imported H5AD annotations in their native representation when that expresses the source contract.
3. **Separate decoding from meaning.** The parser owns file lifetime, projection, null spellings, batching and error context. Bind those options to the schema's field policy. The schema owner checks facts such as positive IDs, finite required coordinates or unique feature names. Where a field carries scientific information in its spelling, retain that spelling until it is interpreted. For example, CosMx uses the printed precision of a position to bound registration uncertainty.
4. **Carry the established facts forward.** Give private functions the precise type their producer established. Check a JSON key before narrowing a dictionary to a type requiring it. A frozen record can carry a mutable owned frame. State which stage may update that frame. Preserve Arrow-backed annotations until the AnnData or spatial model boundary and avoid a second conversion merely to pass between helpers.
5. **Check relationships where both sides meet.** A schema can establish that a barcode is valid on one side. Its membership in a second artifact belongs to alignment. Likewise, a finite pitch and an invertible image registration can compose to a singular transform in float64. Check the composed mapping at the stage that produces it. Finish filtering, aggregation, scaling and construction of target elements before linking each table once. Use an independent target index when available. With lazy labels, use known bounds such as the inspected dtype and document the extent of the check.

### A worked example

In MCMICRO, `_layout.py` pairs each quantification file with its segmentation target and distinguishes whole slides from tissue microarrays. `_metadata.read_markers()` asks `_schemas.marker_types()` for the panel's column types before reading the marker rows. `_schemas.quantification_columns()` then uses the marker panel and the quantification header to assign each column its Arrow type and its role as expression, morphology or text. A new text annotation therefore cannot silently become an expression feature. `_tables.py` can pass those types to Arrow, build sparse expression in batches, and align it to the marker order. The raster builder supplies the target grid and label dtype. The table builder has the evidence it needs to check representability and link each quantification to its target. [MCMICRO format](mcmicro.md)

For a fixed schema, start with classic Visium's `_schemas.POSITION_SOURCE_SCHEMA` and its headered and positional fixtures. For multiple versions, Xenium's `_layout.TabularSource` carries the file, storage kind and schema for an artifact. Its element readers consume that selection directly. [Visium format](visium.md) · [Xenium format](xenium.md)

Plain and gzip transcript partitions share one decoder for complete physical CSV records. It checks quote pairing and line endings, preserves a literal U+FEFF in data, and enlarges its normal 1 MiB Arrow parser block when a record requires it. Partition boundaries therefore do not determine whether a long transcript row can be read. Memory must still accommodate the whole record. Count-matrix readers have separate bounded-batch contracts.

## Acceptance for a new reader

| Where to check | Evidence to include |
| --- | --- |
| Mirrored unit tests such as `test_layout.py` and `test_schemas.py` | Exercise supported layout alternatives, source selection, column order, extra and missing fields, exact identifier widths, nulls and semantic constraints at their owning boundary. Strictly type each private producer and consumer. |
| Product tests | Use fixed expected coordinates, units, values, categories and sparse order. Check lazy products at construction and on a computed selection. Compare the same records or pixels across supported encodings and chunk boundaries. Include a damaged later member in archive fixtures to expose readers that silently return partial results. Put data-dependent checks in the task that owns the data. |
| Public workflow and CLI | Supply a small synthetic input per supported layout or version. Assert element names, axis and feature order, transforms, final regions and IDs, options and provenance together. Include the failure that would result if two files disagreed. |
| Format documentation and performance | Record the source evidence for scientific choices, missing-data behavior and ownership. For claims about copying, graph size or I/O, measure identical inputs and verify numerical equivalence. |

Use the shared foundations for path containment and error context, typed CSV blocks, raster I/O, supported 10x counts, sparse normalization, final table linkage and provenance. Vendor names, coordinate frames, geometry drop semantics and expression vocabularies stay with their format. [Paths](dataset_paths.md) · [Raster options](raster_options.md) · [Calibration](calibration.md) · [Table contracts](table_contracts.md)
