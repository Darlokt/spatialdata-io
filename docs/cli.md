# Command-line reference

Run the CLI with the Python environment where you installed spatialdata-io.
List the available commands or inspect a reader's options with:

```bash
python -m spatialdata_io --help
python -m spatialdata_io xenium --help
```

Dataset commands take an input directory and write a SpatialData Zarr store.
The `generic` command takes a single file and can also append an element to a
store. For the expected input layout and interpretation of the output, see the
[reader guides](readers/index.md). For datasets stored with git-annex or DVC,
see [dataset paths and symlinks](user/paths.md).

```{eval-rst}
.. click:: spatialdata_io.__main__:cli
    :prog: python -m spatialdata_io
    :nested: full
```
