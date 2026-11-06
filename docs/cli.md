# CLI

This section documents the Command Line Interface (CLI) for the `spatialdata_io` package.

Every dataset command accepts `--path-policy [strict|relaxed]` (default
`strict`). Use `relaxed` for git-annex or DVC datasets whose files, including
relative file options, are symlinks to targets outside the input directory; see
[Dataset paths and symlinks](dataset_paths.md).

```{eval-rst}
.. click:: spatialdata_io.__main__:cli
    :prog: spatialdata_io
    :nested: full
```
