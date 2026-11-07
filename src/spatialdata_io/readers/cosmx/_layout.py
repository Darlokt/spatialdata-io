"""Discover one CosMx flat-file export and its numerically ordered FOV rasters.

Supported exports (the 2021 flat-file release and AtoMx flat-file exports; no
version label is read)::

    export/
    ├── <prefix>_exprMat_file.csv           # count matrix
    ├── <prefix>_metadata_file.csv          # cell metadata
    ├── <prefix>_fov_positions_file.csv     # optional FOV positions
    ├── <prefix>_tx_file.csv                # transcripts, required when requested
    ├── CellComposite/<any>_F<fov>.jpg      # optional; .jpeg, .png, .tif or .tiff
    └── CellLabels/<any>_F<fov>.tif         # optional; .tiff

Every flat file may be gzip-compressed instead (``.csv.gz``), but not exist in
both spellings. The file prefix is the caller's ``file_prefix`` or, by default,
the prefix of the only ``*_exprMat_file.csv[.gz]`` in the directory; it may
contain any character except path separators and NUL, including spaces. The
element names derived from it are chosen by the reader, not here.

FOV numbers come from the ``_F<number>`` raster suffix (positive, any number of
digits, compared numerically, so ``F002`` and ``F2`` are the same FOV) and are
returned in numeric order. Dot files are ignored; any other unexpected file in
a raster directory fails. An empty ``CellComposite`` directory is accepted; a
``CellLabels`` directory must contain at least one raster, because its presence
links the table to labels. Polygon, overlay, compartment and morphology files
are not read.

The vendor `human-brain release README
<https://www.brukerspatialbiology.com/wp-content/uploads/2023/09/SMI-ReadMe-BETA_humanBrainRelease.html>`_
describes the flat files and their local and global frames.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactDirectory,
    ArtifactFile,
    DatasetRoot,
    list_directory,
    normalize_dataset_root,
    optional_metadata_directory,
    require_metadata_file,
)
from spatialdata_io.readers.cosmx._schemas import COUNTS, IMAGES, LABELS, METADATA, POSITIONS, TRANSCRIPTS

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.paths import PathPolicy
    from spatialdata_io.readers.cosmx._schemas import Artifact

_SUFFIXES: dict[Artifact, str] = {
    COUNTS: "exprMat_file",
    METADATA: "metadata_file",
    POSITIONS: "fov_positions_file",
    TRANSCRIPTS: "tx_file",
}
_EXTENSIONS = (".csv", ".csv.gz")
_COUNTS_NAME = re.compile(rf"(?P<prefix>.+)_{_SUFFIXES[COUNTS]}\.csv(?:\.gz)?")
_FOV_FILE = re.compile(r".*_F(?P<fov>[0-9]+)\.(?P<suffix>[A-Za-z]+)")
_IMAGE_SUFFIXES = frozenset({"jpg", "jpeg", "png", "tif", "tiff"})
_LABEL_SUFFIXES = frozenset({"tif", "tiff"})
_ROOT_CONTEXT = ReaderErrorContext("cosmx", "dataset root")


@dataclass(frozen=True, slots=True)
class FovFile:
    """Pair one positive FOV number with its validated raster file."""

    fov: int
    path: ArtifactFile


@dataclass(frozen=True, slots=True)
class CellLabels:
    """A present ``CellLabels`` directory and its nonempty, FOV-ordered rasters."""

    directory: ArtifactDirectory
    files: tuple[FovFile, ...]


@dataclass(frozen=True, slots=True)
class CosmxLayout:
    """Describe the artifacts of one export; geometry and schemas are read later."""

    file_prefix: str
    counts: ArtifactFile
    metadata: ArtifactFile
    fov_positions: ArtifactFile | None
    transcripts: ArtifactFile | None
    images: tuple[FovFile, ...]
    labels: CellLabels | None


def _error(path: Path, artifact: Artifact, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a layout conflict with its reader-owned artifact role."""
    return ReaderFormatError(context=ReaderErrorContext("cosmx", artifact), path=path, found=found, expected=expected)


def validate_file_prefix(file_prefix: object) -> None:
    """Reject a caller prefix that cannot name files directly inside the export, before any I/O."""
    if file_prefix is None:
        return
    if not isinstance(file_prefix, str):
        message = f"file_prefix must be a string or None; found {type(file_prefix).__name__}."
        raise TypeError(message)
    if not file_prefix or any(character in file_prefix for character in "/\\\x00"):
        message = f"file_prefix must be nonempty and cannot contain path separators or NUL; found {file_prefix!r}."
        raise ValueError(message)


def _infer_file_prefix(root: DatasetRoot, names: tuple[str, ...]) -> str:
    """Infer the file prefix from the single count matrix."""
    matches = [match for name in names if (match := _COUNTS_NAME.fullmatch(name)) is not None]
    if len(matches) != 1:
        raise _error(
            root.path,
            COUNTS,
            found=f"{len(matches)} count matrices",
            expected="exactly one <prefix>_exprMat_file.csv[.gz]; pass file_prefix otherwise",
        )
    return matches[0]["prefix"]


def _present(root: DatasetRoot, names: tuple[str, ...], prefix: str, artifact: Artifact) -> str | None:
    """Return the one present plain or gzip-compressed spelling, rejecting both."""
    present = [name for extension in _EXTENSIONS if (name := f"{prefix}_{_SUFFIXES[artifact]}{extension}") in names]
    if len(present) > 1:
        raise _error(root.path, artifact, found=f"files {present!r}", expected="one plain or compressed file")
    return present[0] if present else None


def _required(root: DatasetRoot, names: tuple[str, ...], prefix: str, artifact: Artifact) -> ArtifactFile:
    """Resolve a required flat file; a missing one fails naming the plain spelling."""
    name = _present(root, names, prefix, artifact) or f"{prefix}_{_SUFFIXES[artifact]}.csv"
    return require_metadata_file(root, name, context=ReaderErrorContext("cosmx", artifact))


def _optional(root: DatasetRoot, names: tuple[str, ...], prefix: str, artifact: Artifact) -> ArtifactFile | None:
    """Resolve an optional flat file when present."""
    name = _present(root, names, prefix, artifact)
    return None if name is None else require_metadata_file(root, name, context=ReaderErrorContext("cosmx", artifact))


def _fov_files(
    root: DatasetRoot, name: str, artifact: Artifact, suffixes: frozenset[str]
) -> tuple[ArtifactDirectory, tuple[FovFile, ...]] | None:
    """Return a present raster directory with one raster per FOV in numeric order."""
    context = ReaderErrorContext("cosmx", artifact)
    directory = optional_metadata_directory(root, name, context=context)
    if directory is None:
        return None
    files: dict[int, ArtifactFile] = {}
    for entry in list_directory(directory, context=context):
        if entry.name.startswith("."):
            continue
        match = _FOV_FILE.fullmatch(entry.name)
        if match is None or match["suffix"].lower() not in suffixes or int(match["fov"]) < 1:
            raise _error(
                entry,
                artifact,
                found=f"file {entry.name!r}",
                expected=f"<prefix>_F<positive FOV>.<{'|'.join(sorted(suffixes))}>",
            )
        fov = int(match["fov"])
        if fov in files:
            raise _error(entry, artifact, found=f"two files for FOV {fov}", expected="one raster per FOV")
        files[fov] = require_metadata_file(root, f"{name}/{entry.name}", context=context)
    return directory, tuple(FovFile(fov, files[fov]) for fov in sorted(files))


def _labels(root: DatasetRoot) -> CellLabels | None:
    """Return the label rasters, rejecting a present directory without any."""
    found = _fov_files(root, "CellLabels", LABELS, _LABEL_SUFFIXES)
    if found is None:
        return None
    directory, files = found
    if not files:
        raise _error(directory, LABELS, found="no label rasters", expected="a nonempty CellLabels directory")
    return CellLabels(directory, files)


def discover_layout(
    path: str | Path, *, file_prefix: str | None, include_transcripts: bool, path_policy: PathPolicy = "strict"
) -> CosmxLayout:
    """Resolve every requested artifact once, without reading rows or pixels.

    ``file_prefix`` must already have passed :func:`validate_file_prefix`.
    """
    root = normalize_dataset_root(path, context=_ROOT_CONTEXT, path_policy=path_policy)
    names = tuple(entry.name for entry in list_directory(root.path, context=_ROOT_CONTEXT))
    prefix = _infer_file_prefix(root, names) if file_prefix is None else file_prefix

    counts = _required(root, names, prefix, COUNTS)
    metadata = _required(root, names, prefix, METADATA)
    positions = _optional(root, names, prefix, POSITIONS)
    transcripts = _required(root, names, prefix, TRANSCRIPTS) if include_transcripts else None
    images = _fov_files(root, "CellComposite", IMAGES, _IMAGE_SUFFIXES)
    return CosmxLayout(
        prefix, counts, metadata, positions, transcripts, () if images is None else images[1], _labels(root)
    )
