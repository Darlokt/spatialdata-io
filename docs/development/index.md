# Development

For your first checkout, follow the [contributing guide](../contributing.md) to
install the tools and run the tests. If you need a real instrument dataset,
[start with one registered test dataset](datasets.md).

## Change a reader

[Add or update a reader](readers.md) covers the public Python and CLI contracts,
module boundaries, tests and documentation. [How the Visium reader works](visium-walkthrough.md)
follows one dataset through discovery, calibration, count decoding and linkage.

```{toctree}
:hidden:

../contributing
../contributing_io
readers
visium-walkthrough
```

## Implement reader outputs

Use [the shared utilities](contracts.md) for file discovery, errors, metadata and
provenance. Consult [rasters and geometry](rasters.md) for lazy pixel access,
transformations and invalid polygons. Use [tables and transcript points](tables.md)
for sparse counts, Arrow schemas, AnnData positions and region linkage.

```{toctree}
:hidden:

contracts
rasters
tables
```

## Check your change

[Download or register test data](datasets.md) for a public workflow, then use
[the testing guide](testing.md) to check the format contract.

```{toctree}
:hidden:

datasets
testing
```
