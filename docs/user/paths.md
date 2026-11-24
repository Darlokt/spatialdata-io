# Read a dataset with symlinks

Pass the dataset directory shown in the [reader guide](../readers/index.md) as
the input. By default (`path_policy="strict"`), discovered files, directories,
and relative file options must resolve inside that directory, including after
following symlinks. An error that mentions an outside target can occur when
git-annex or DVC keeps the data in a separate cache.

For such a dataset, allow linked targets outside the input directory:

```python
from spatialdata_io import visium

sdata = visium("dataset/outs", path_policy="relaxed")
```

The CLI equivalent is:

```bash
python -m spatialdata_io visium --input dataset/outs --output sample.zarr --path-policy relaxed
```

Both modes resolve relative file options against the dataset directory. A
relative option or path in the dataset's own metadata cannot use `..` to escape
it. An explicit absolute file option can point elsewhere in either mode. With
relaxed paths, sample names and file types come from the names in the dataset,
including when a cache target has no suffix. Missing required files, dangling
links and unreadable input directories still raise errors. Readers skip macOS
AppleDouble companion files (`._*`).

## Keep multi-file OME-TIFF images together

Some images, including certain Xenium morphology exports, store their channels
in separate OME-TIFF files that refer to each other by name. The TIFF reader
looks for the other files beside the **real target** of the first file. Separate
symlinks into a content-addressed cache can scatter those targets and cause a
`RasterFormatError` reporting a missing TIFF source.

Materialize the members as regular files together (for example, use
`git annex unlock` or a DVC copy link type). Alternatively, link the entire
directory with the original filenames intact and use `path_policy="relaxed"`.
