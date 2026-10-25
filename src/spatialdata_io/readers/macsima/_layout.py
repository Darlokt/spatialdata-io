"""Discover processed MACSima folders as immutable stacks of candidate TIFFs."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from spatialdata import sanitize_name

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    ArtifactDirectory,
    ArtifactFile,
    DatasetRoot,
    normalize_dataset_root,
    require_directory,
    require_file,
    sorted_paths,
)

if TYPE_CHECKING:
    from pathlib import Path

type ParsingStyle = Literal["processed_single_folder", "processed_multiple_folders"]

_PARSING_STYLES: tuple[ParsingStyle, ...] = ("processed_single_folder", "processed_multiple_folders")
_CONTEXT = ReaderErrorContext("macsima", "layout")
_TIFF_SUFFIXES = frozenset({".tif", ".tiff"})
# macOS archive metadata; its "._name.tif" entries are not TIFF files.
_ARCHIVE_METADATA = "__MACOSX"


@dataclass(frozen=True, slots=True)
class StackFiles:
    """Candidate channel files of one processed folder, in lexical path order.

    ``name`` is the folder name made valid for SpatialData; it names the stack's
    elements and coordinate system. ``directory`` is the folder itself.
    """

    name: str
    directory: ArtifactDirectory
    files: tuple[ArtifactFile, ...]


@dataclass(frozen=True, slots=True)
class MacsimaLayout:
    """Every stack of one invocation; both parsing styles share this shape."""

    root: DatasetRoot
    stacks: tuple[StackFiles, ...]


def validate_layout_options(parsing_style: ParsingStyle, exclude_folder_names: Sequence[str]) -> frozenset[str]:
    """Validate the parsing style and own the exact folder names it may exclude."""
    if parsing_style not in _PARSING_STYLES:
        message = f"parsing_style must be one of {list(_PARSING_STYLES)}, not {parsing_style!r}."
        raise ValueError(message)
    if isinstance(exclude_folder_names, str | bytes) or not isinstance(exclude_folder_names, Sequence):
        message = "exclude_folder_names must be a sequence of folder names."
        raise TypeError(message)
    if any(not isinstance(name, str) for name in exclude_folder_names):
        message = "exclude_folder_names must contain only strings."
        raise TypeError(message)
    if not all(exclude_folder_names):
        message = "exclude_folder_names must not contain empty names."
        raise ValueError(message)
    if exclude_folder_names and parsing_style == "processed_single_folder":
        message = "exclude_folder_names requires parsing_style='processed_multiple_folders'."
        raise ValueError(message)
    return frozenset(exclude_folder_names)


def _visible(name: str) -> bool:
    return not name.startswith(".") and name != _ARCHIVE_METADATA


def _is_candidate(path: Path, directory: ArtifactDirectory) -> bool:
    """Accept a regular file with a TIFF suffix and no hidden path component."""
    return (
        path.suffix.lower() in _TIFF_SUFFIXES
        and all(_visible(part) for part in path.relative_to(directory).parts)
        and path.is_file()
    )


def _candidate_files(directory: ArtifactDirectory) -> tuple[ArtifactFile, ...]:
    """Return the candidate TIFFs below a folder in lexical path order."""
    paths = sorted_paths(directory.rglob("*"), context=_CONTEXT, search_path=directory)
    return tuple(require_file(path, context=_CONTEXT) for path in paths if _is_candidate(path, directory))


def _require_unique_names(stacks: Sequence[StackFiles], root: DatasetRoot) -> None:
    """Reject folders whose element names collide after sanitizing."""
    # SpatialData element names must be unique ignoring case.
    counts = Counter(stack.name.casefold() for stack in stacks)
    colliding = sorted({stack.name for stack in stacks if counts[stack.name.casefold()] > 1})
    if colliding:
        raise ReaderFormatError(
            context=_CONTEXT,
            path=root,
            found=f"folders that become the colliding element names {colliding}",
            expected="folder names that remain distinct as SpatialData element names",
        )


def _single_folder(root: DatasetRoot) -> tuple[StackFiles, ...]:
    files = _candidate_files(root)
    if not files:
        raise ReaderFormatError(
            context=_CONTEXT, path=root, found="no .tif or .tiff files", expected="processed MACSima channel TIFFs"
        )
    return (StackFiles(sanitize_name(root.name), root, files),)


def _multiple_folders(root: DatasetRoot, exclude_folder_names: frozenset[str]) -> tuple[StackFiles, ...]:
    folders = sorted_paths(
        (entry for entry in root.iterdir() if _visible(entry.name) and entry.is_dir()),
        context=_CONTEXT,
        search_path=root,
    )
    unknown = exclude_folder_names - {folder.name for folder in folders}
    if unknown:
        message = f"exclude_folder_names {sorted(unknown)} match no folder in {str(root)!r}."
        raise ValueError(message)

    stacks: list[StackFiles] = []
    for folder in folders:
        if folder.name in exclude_folder_names:
            continue
        directory = require_directory(folder, context=_CONTEXT)
        files = _candidate_files(directory)
        if not files:
            warnings.warn(f"No TIFF files found in {str(folder)!r}; skipping this folder.", UserWarning, stacklevel=4)
            continue
        stacks.append(StackFiles(sanitize_name(folder.name), directory, files))

    if not stacks:
        raise ReaderFormatError(
            context=_CONTEXT,
            path=root,
            found="no included subfolder with .tif or .tiff files",
            expected="at least one processed MACSima folder",
        )
    _require_unique_names(stacks, root)
    return tuple(stacks)


def discover_layout(
    path: str | Path, *, parsing_style: ParsingStyle, exclude_folder_names: frozenset[str]
) -> MacsimaLayout:
    """Resolve the root once and collect the candidate files of every stack.

    A single-folder layout stacks every TIFF below the root. A multiple-folder
    layout makes one stack per immediate subfolder; folders named exactly in
    ``exclude_folder_names`` are omitted and folders without TIFFs are skipped
    with a warning. Paths beginning with ``.`` and ``__MACOSX`` are ignored.
    """
    root = normalize_dataset_root(path, context=_CONTEXT)
    if parsing_style == "processed_single_folder":
        return MacsimaLayout(root, _single_folder(root))
    return MacsimaLayout(root, _multiple_folders(root, exclude_folder_names))
