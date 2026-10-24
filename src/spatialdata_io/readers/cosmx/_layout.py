"""Discover one CosMx flat-file export and its numerically ordered FOV rasters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactDirectory,
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    optional_directory,
    require_metadata_file,
    sorted_paths,
)

if TYPE_CHECKING:
    from pathlib import Path

_COUNTS = "exprMat_file"
_EXTENSIONS = (".csv", ".csv.gz")
_COUNTS_NAME = re.compile(rf"(?P<dataset>.+)_{_COUNTS}\.csv(?:\.gz)?")
_FOV_FILE = re.compile(r".*_F(?P<fov>[0-9]+)\.(?P<suffix>[A-Za-z]+)")
_IMAGE_SUFFIXES = frozenset({"jpg", "jpeg", "png", "tif", "tiff"})
_LABEL_SUFFIXES = frozenset({"tif", "tiff"})


@dataclass(frozen=True, slots=True)
class FovFile:
    """Pair one positive FOV number with its validated raster file."""

    fov: int
    path: ArtifactFile


@dataclass(frozen=True, slots=True)
class CosmxLayout:
    """Describe the artifacts of one export; geometry and schemas are read later."""

    root: DatasetRoot
    dataset_id: str
    counts: ArtifactFile
    metadata: ArtifactFile
    fov_positions: ArtifactFile | None
    transcripts: ArtifactFile | None
    images: tuple[FovFile, ...]
    labels: tuple[FovFile, ...]


def _error(path: Path, *, artifact: str, found: str, expected: str) -> ReaderFormatError:
    """Describe a layout conflict with its reader-owned artifact role."""
    return ReaderFormatError(context=ReaderErrorContext("cosmx", artifact), path=path, found=found, expected=expected)


def _infer_dataset_id(root: DatasetRoot, names: tuple[str, ...]) -> str:
    """Infer the file prefix from the single count matrix."""
    matches = [match for name in names if (match := _COUNTS_NAME.fullmatch(name)) is not None]
    if len(matches) != 1:
        raise _error(
            root,
            artifact="count matrix",
            found=f"{len(matches)} count matrices",
            expected="exactly one <dataset_id>_exprMat_file.csv[.gz]; pass dataset_id otherwise",
        )
    return matches[0]["dataset"]


def _validate_dataset_id(dataset_id: str) -> None:
    """Reject unusable caller prefixes before filesystem access."""
    if not isinstance(dataset_id, str):
        message = "dataset_id must be a string or None."
        raise TypeError(message)
    if not dataset_id.strip() or any(character in dataset_id for character in "/\\\x00"):
        message = "dataset_id must be nonempty and cannot contain path separators."
        raise ValueError(message)


def _present(root: DatasetRoot, names: tuple[str, ...], dataset_id: str, suffix: str) -> str | None:
    """Return the one present plain or gzip-compressed spelling, rejecting both."""
    present = [name for extension in _EXTENSIONS if (name := f"{dataset_id}_{suffix}{extension}") in names]
    if len(present) > 1:
        raise _error(root, artifact=suffix, found=f"files {present!r}", expected="one plain or compressed file")
    return present[0] if present else None


def _required(root: DatasetRoot, names: tuple[str, ...], dataset_id: str, suffix: str) -> ArtifactFile:
    """Resolve a required flat file; a missing one fails naming the plain spelling."""
    name = _present(root, names, dataset_id, suffix) or f"{dataset_id}_{suffix}.csv"
    return require_metadata_file(root, name, context=ReaderErrorContext("cosmx", suffix))


def _optional(root: DatasetRoot, names: tuple[str, ...], dataset_id: str, suffix: str) -> ArtifactFile | None:
    """Resolve an optional flat file when present."""
    name = _present(root, names, dataset_id, suffix)
    return None if name is None else require_metadata_file(root, name, context=ReaderErrorContext("cosmx", suffix))


def _fov_files(root: DatasetRoot, name: str, suffixes: frozenset[str]) -> tuple[FovFile, ...]:
    """Return one raster per FOV in numeric order, rejecting unsupported or repeated files."""
    context = ReaderErrorContext("cosmx", name)
    directory: ArtifactDirectory | None = optional_directory(root / name, context=context)
    if directory is None:
        return ()
    files: dict[int, ArtifactFile] = {}
    for entry in sorted_paths(directory.iterdir(), context=context, search_path=directory):
        if entry.name.startswith("."):
            continue
        match = _FOV_FILE.fullmatch(entry.name)
        if match is None or match["suffix"].lower() not in suffixes or int(match["fov"]) < 1:
            raise _error(
                entry,
                artifact=name,
                found=f"file {entry.name!r}",
                expected=f"<prefix>_F<positive FOV>.<{'|'.join(sorted(suffixes))}>",
            )
        fov = int(match["fov"])
        if fov in files:
            raise _error(entry, artifact=name, found=f"two files for FOV {fov}", expected="one raster per FOV")
        files[fov] = require_metadata_file(root, f"{name}/{entry.name}", context=context)
    return tuple(FovFile(fov, files[fov]) for fov in sorted(files))


def discover_layout(path: str | Path, *, dataset_id: str | None, include_transcripts: bool) -> CosmxLayout:
    """Resolve every requested artifact once, without reading rows or pixels."""
    if dataset_id is not None:
        _validate_dataset_id(dataset_id)
    root = normalize_dataset_root(path, context=ReaderErrorContext("cosmx", "dataset root"))
    context = ReaderErrorContext("cosmx", "dataset root")
    names = tuple(entry.name for entry in sorted_paths(root.iterdir(), context=context, search_path=root))
    prefix = _infer_dataset_id(root, names) if dataset_id is None else dataset_id

    return CosmxLayout(
        root,
        prefix,
        _required(root, names, prefix, _COUNTS),
        _required(root, names, prefix, "metadata_file"),
        _optional(root, names, prefix, "fov_positions_file"),
        _required(root, names, prefix, "tx_file") if include_transcripts else None,
        _fov_files(root, "CellComposite", _IMAGE_SUFFIXES),
        _fov_files(root, "CellLabels", _LABEL_SUFFIXES),
    )
