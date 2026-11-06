# Dataset paths and symlinks

Every reader that takes a dataset directory exposes the same option,
`path_policy`, and every command exposes it as `--path-policy`. It decides how
symlinks found inside the dataset directory are treated.

| Value | Meaning |
| --- | --- |
| `"strict"` (default) | Every artifact found in the dataset, and every relative file option, must resolve, symlinks followed, to a file or directory inside the dataset directory. Otherwise the reader raises `ReaderFormatError`. |
| `"relaxed"` | These artifacts may also be symlinks to targets stored elsewhere, as in git-annex and DVC datasets whose files link into an object store or cache. |

```python
from spatialdata_io import visium

sdata = visium("dataset/outs", path_policy="relaxed")
```

```bash
spatialdata-io visium --input dataset/outs --output sample.zarr --path-policy relaxed
```

Rules that hold in both modes:

- A path named by the dataset itself, for example an image entry in Xenium's
  `experiment.xenium`, must not leave the dataset directory: an absolute path
  elsewhere or a `..` escape is rejected.
- A relative file option, such as `fullres_image_file`, `cell_table_file` or
  ISS's `table_file`, may not leave the dataset directory with `..`; under
  `"relaxed"` it may be a symlink to a target elsewhere, exactly like a
  discovered artifact.
- An absolute file option may point anywhere under both values.
- File options keep the name you gave, and a relaxed read keeps each
  discovered artifact's own name, suffix and directory. Annex and DVC targets
  are named by content hashes (DVC cache entries have no suffix), so names that
  identify samples, channels or file formats always come from the dataset or
  from you, never from the target.
- Missing files, dangling symlinks and files of the wrong kind fail as before.

## Known limitation: multi-file OME-TIFF sets

A multi-file OME-TIFF set stores one image across several files that reference
each other by name, for example Xenium's `morphology_focus/ch0000_*.ome.tif`
files. `tifffile` always opens the selected file at its real location and
looks for the other files of the set beside that real file. When the members
are symlinks to separate targets, as git-annex and DVC create, the other files
are not beside the target. The read then fails with `RasterFormatError`
("OME metadata references a non-file TIFF source") instead of returning
zero-filled channels. Such sets are not supported with `"relaxed"`; unlock or
check out the set as regular files (for example `git annex unlock` or a DVC
copy link type) to read it. Single-file TIFF, OME-TIFF, PNG and JPEG images are
not affected.

Compared with earlier releases, `visium`, `visium_hd`, `xenium` and `macsima`
now also reject out-of-root symlinks by default; they previously accepted them.
Pass `path_policy="relaxed"` to keep reading such datasets.
