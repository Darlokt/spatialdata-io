# Reader path boundary

This private package provides deterministic local-filesystem primitives for
`spatialdata-io` readers. It centralizes canonical path representation,
containment, artifact-kind validation, deterministic candidate ordering, and
ambiguity handling without interpreting any vendor format.

Generic reader errors live in `spatialdata_io.readers._utils.errors`. The image,
table, and path packages share that error boundary without depending on one
another.

## Responsibilities

The package owns:

- canonical existing dataset roots and their path policy;
- required and optional regular-file and directory validation;
- distinct caller and dataset (metadata) path-origin rules;
- deterministic directory listing, without macOS AppleDouble (`._*`)
  companions, and unique selection;
- precise standard path exceptions and contextual format failures; and
- metadata-only filesystem work with no content loading.

A reader owns:

- filenames, suffixes, patterns, recursion, pairing, and directory meanings;
- version detection and required-artifact policy;
- filename-derived identifiers and other lexical semantics;
- interpretation of file contents, coordinates, geometry, and transformations;
- resource ownership after validation; and
- its immutable version-specific layout representation.

The package supports local `str` and `Path` values only. URLs, fsspec stores,
generic discovery callbacks, vendor manifests, and layout base classes are
outside this boundary.

## Canonical and kind-marked path representation

The validation functions return allocation-free nominal types around ordinary
canonical absolute `Path` objects:

```python
ArtifactFile = NewType("ArtifactFile", Path)
ArtifactDirectory = NewType("ArtifactDirectory", Path)


@dataclass(frozen=True, slots=True)
class DatasetRoot:
    path: ArtifactDirectory
    path_policy: PathPolicy  # Literal["strict", "relaxed"]
```

`ArtifactFile` records successful regular-file validation and
`ArtifactDirectory` records successful directory validation. Both remain `Path`
subtypes to static type checkers and are the identical `Path` objects at
runtime: no wrapper object, copying, or conversion is introduced.
`DatasetRoot` pairs the canonical root directory with the path policy chosen
for one reader call. It is the single representation of that policy: every
root-relative validator reads it from the root, so readers never pass the
policy to helpers. Use `root.path` wherever a directory path is needed.

Only this package's validation functions construct these values. Reader code
imports them for annotations and stores returned values; it must not call the
`NewType` constructors to relabel an unvalidated `Path`.

Dataset validators of a strict root and the root-free validators return the
canonical resolved target. This gives path-backed delayed tasks one stable
source spelling and prevents a validated symlink alias from later being
redirected elsewhere. Caller validators, and every validator of a relaxed root,
return the resolved parent directory joined with the given final name instead;
see the origin rules below. Readers that derive
scientific identifiers from a candidate's lexical filename must record that
identifier before validation or store it separately from the canonical I/O
path.

The marker records validation at the reader boundary; it does not pin the
filesystem entry or guarantee that another process cannot replace or remove it
later. Downstream consumers still own their ordinary open/read error behavior.
Repeating a kind check cannot remove that time-of-check/time-of-use limitation.

Aliases remain distinct during candidate discovery. Two names pointing at one
target can still make a vendor layout ambiguous; canonicalization happens only
after the reader selects and validates one candidate.

## Path policy

A public reader exposes `path_policy: Literal["strict", "relaxed"] = "strict"`
and passes it once to `normalize_dataset_root()`, which validates it before any
filesystem access (`TypeError` for a non-string, `ValueError` for another
value) and stores it in the returned `DatasetRoot`. Every artifact that a reader
resolves from, or discovers under, that root obeys it:

- `"strict"`: the artifact must resolve, symlinks followed, to a target inside
  the dataset root; otherwise `ReaderFormatError` is raised and its message
  names the relaxed policy.
- `"relaxed"`: the artifact may be a symlink whose target lies anywhere, as in
  git-annex and DVC datasets.

Lexical containment applies in both modes, and kind checks, missing and
dangling targets behave identically in both modes. The exact rules per origin
are listed below.

Libraries that resolve symlinks themselves still read the target. In
particular, tifffile opens linked files of a multi-file OME-TIFF beside the real
path of the opened file, so the shared TIFF inspection checks linked files
there and rejects a scattered set instead of reading zero-filled data.

## Dataset, caller, and root-free paths

Every path under a dataset root, whether a reader-owned name, a discovered
entry, or a value read from metadata, uses `require_metadata_file()`,
`require_metadata_directory()`, `optional_metadata_file()`, or
`optional_metadata_directory()`:

```python
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.paths import (
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    require_metadata_file,
)

context = ReaderErrorContext(reader="example", artifact="counts")
root: DatasetRoot = normalize_dataset_root("dataset", context=context, path_policy="strict")
counts: ArtifactFile = require_metadata_file(root, "counts.h5", context=context)
```

`require_file()` and `require_directory()` remain for paths without a dataset
root: a single caller-selected file of a root-free reader, or a reader-created
directory such as an archive extraction target. They never apply a policy.

Paths have three origins:

| Origin | Lexical check | Target, strict root | Target, relaxed root | Returned path |
| --- | --- | --- | --- | --- |
| Dataset (reader-owned name, discovered entry, metadata value; no `~` expansion) | inside the root | inside the root | anywhere | strict: canonical target; relaxed: resolved parent + own name |
| Relative caller path (`~` expanded) | inside the root | inside the root | anywhere | resolved parent + given name |
| Absolute caller path (`~` expanded) | none | anywhere | anywhere | resolved parent + given name |

"Resolved parent + name" keeps the final name the dataset or the caller gave,
because git-annex and DVC targets are named by content hashes (DVC cache entries
have no suffix). Names and suffixes that identify samples, channels or formats
therefore never come from a link target, and paths joined to a returned
directory stay lexically inside the root. A strict rejection names the relaxed
policy in its message.

Paths that require containment are checked lexically before filesystem access,
then checked again after strict resolution to reject symlink escapes. Explicit
traversal and external metadata paths therefore fail consistently even when the
outside target is missing or inaccessible.

Origin and kind validation are atomic:

```python
from spatialdata_io.readers._utils.paths import (
    require_caller_directory,
    require_caller_file,
    require_metadata_directory,
    require_metadata_file,
)

microscopy = require_caller_file(root, image_override, context=image_context)
analysis = require_caller_directory(root, output_override, context=output_context)
counts = require_metadata_file(root, manifest.counts, context=counts_context)
store = require_metadata_directory(root, manifest.store, context=store_context)
```

These functions resolve, contain, and kind-check once. Downstream private
parsers receive a canonical `ArtifactFile` or `ArtifactDirectory` and do not
repeat origin or kind validation.

## Reader layouts and downstream consumers

Reader-owned frozen layouts should retain the most precise returned type:

```python
from dataclasses import dataclass

from spatialdata_io.readers._utils.paths import (
    ArtifactDirectory,
    ArtifactFile,
    DatasetRoot,
)


@dataclass(frozen=True, slots=True)
class ExampleLayout:
    root: DatasetRoot
    counts_h5: ArtifactFile
    matrix_directory: ArtifactDirectory | None
    morphology: ArtifactFile
```

The values are directly consumable by APIs that accept `Path` or
`os.PathLike[str]`; callers do not unwrap or reconstruct them:

```python
import scanpy as sc

from spatialdata_io.readers._utils.image import inspect_tiff, read_tiff

adata = sc.read_10x_h5(layout.counts_h5)
metadata = inspect_tiff(layout.morphology, context=context)
image = read_tiff(metadata, series=0, level=0)
```

The direct Scanpy call demonstrates ecosystem compatibility. Vendor readers
should use the focused shared count-source boundary once it is available so
sparse-array normalization and source error context remain centralized.

Likewise, `root.path` can be passed directly to a directory consumer such
as `scanpy.read_10x_mtx()`. Focused shared source utilities should accept the
nominal artifact type and trust its path contract instead of resolving or
kind-checking it again. Broad direct-entry utilities such as the existing image
inspection functions may retain their own input normalization while accepting
the marker as an ordinary `Path`.

The markers do not change execution semantics. TIFF metadata inspection remains
bounded and eager, TIFF pixels remain lazy and native-chunked, PNG/JPEG pixel
decoding remains delayed, and count-table sparsity is owned by the count/table
boundaries rather than this path package.

## Optional artifacts

Optional means that only the final path component may be absent. Its parent must
resolve to an existing directory. This makes malformed layouts explicit:

- lexically escaping path: raise `ReaderFormatError`, even when absent;
- absent leaf under a valid directory: return `None`;
- missing or dangling parent: raise `FileNotFoundError`;
- dangling leaf symlink: raise `FileNotFoundError`;
- wrong or unsupported existing kind: raise the corresponding contextual error.

Readers should validate an optional directory before looking for optional files
inside it. A caller-provided override is required whenever it is not `None`; an
invalid supplied override is never silently skipped.

## Files and directories in discovery

Discovery is intentionally kind-agnostic. Readers may find files, directories,
or both, then apply their format meaning after deterministic selection:

```python
from spatialdata_io.readers._utils.paths import (
    list_directory,
    require_metadata_directory,
    require_unique_path,
)

candidate = require_unique_path(
    (path for path in list_directory(root.path, context=layout_context) if path.name.startswith("analysis-")),
    context=layout_context,
    search_path=root.path,
)
analysis_directory = require_metadata_directory(root, candidate, context=layout_context)
```

`list_directory()` is the only way readers list a directory. It lists inside
its own error handling, so a failure carries reader and directory context
(since Python 3.13 `Path.iterdir()` lists eagerly, before a caller could wrap
its iteration), returns the entries sorted by POSIX spelling, and leaves out
macOS AppleDouble companions (`._<name>`) in every reader. Other hidden
entries are listed; readers decide whether they matter. With
`recursive=True` it walks subdirectories and raises if one cannot be listed,
instead of skipping it silently like `Path.rglob()`. A reader that ignores
hidden paths passes `enter_hidden_directories=False`: subdirectories whose
name starts with `.` are then listed but not entered, so an unreadable cache or
`.git` directory cannot fail its read. Hidden files are listed either way.

Unique selectors consume their reader-filtered iterable once, do not resolve
or filter candidates, and avoid sorting successful zero- or one-candidate
cases. Ambiguities report every candidate in deterministic lexical order.

## Errors

`ReaderErrorContext` identifies the reader, semantic artifact, and optional
already-detected format version. `ReaderFormatError` records that context plus
the found state, expected contract, and optional path. Both are defined in the
shared errors module because table and other non-path utilities use the same
diagnostic contract.

Missing and wrong-kind paths use `FileNotFoundError`, `NotADirectoryError`, or
`IsADirectoryError` with meaningful `errno` and `filename` attributes.
Ambiguity, containment escape, unsupported filesystem entries, and declared
format violations use `ReaderFormatError`. Preserved exceptions receive a note;
converted exceptions retain their original cause.

`RasterFormatError` also lives in the shared errors module but remains a
specialized type for malformed encoded raster storage. Low-level table
representation functions continue to use precise `TypeError` and `ValueError`
contracts rather than wrapping programmer or array-contract failures in a
reader-layout exception.

## Performance and resources

Successful validation performs one strict canonical resolution and one explicit
kind inspection; a relaxed root adds one resolution of the artifact's parent. Additional `lstat` work is limited to optional-leaf handling or
failure diagnosis. Directory validators never scan directory contents.

Creating an artifact marker is an identity call with no wrapper allocation.
The type follow-up adds no filesystem access, file opening, content loading, or
copying.

Unique selectors necessarily snapshot candidate names once. `list_directory()`
sorts the listing directly and returns one immutable tuple. These operations
never store file contents, open handles, or reader state. The package opens no artifact and has no NumPy, SciPy,
AnnData, Dask, Arrow, Scanpy, SpatialData, or raster-library dependency.

## Extension policy

Add a capability only when at least two concrete readers demonstrate the same
filesystem safety or representation contract. Extensions must remain
vendor-neutral, precisely typed, independently tested, and content-free. Do not
add vendor branches, callback discovery, generic resource management,
per-artifact containment switches beside the root's path policy, or a universal
layout schema.
