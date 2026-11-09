"""Discover the channel stacks of a processed MACSima export.

MACS iQ View (MACSiQView) exports processed MACSima data as one single-plane
OME-TIFF per channel image: every stain, bleach and autofluorescence image of
every acquisition cycle of a region of interest (ROI) is its own file. The
public OMAP-10 (software major 0, https://doi.org/10.5281/zenodo.18196366) and
OMAP-23 (major 1, https://doi.org/10.5281/zenodo.18196452) releases keep one
ROI per folder::

    OMAP23_small/
    ├── C-001_S-000_S_APC_R-01_W-D01_ROI-01_A-CD3_C-REA1151.tif
    ├── C-001_S-000_S_DAPI_R-01_W-D01_ROI-01_A-DAPI.tif
    ├── C-002_S-000_S_PE_R-01_W-D01_ROI-01_A-CD279_C-REA1165.tif
    ├── C-004_S-000_S_FITC_R-01_W-D01_ROI-01_A-CD66b_C-REA306.tif
    └── C-015_S-000_B_DAPI_R-01_W-D01_ROI-01_A-DAPI.tif

The file names spell the cycle (``C-015``), the acquisition type (``S``
stain, ``B`` bleach), the fluorochrome, the well (``W-``), the ROI
(``ROI-``), the antigen (``A-``) and the clone (``C-``). Both releases use
this scheme, but the reader never interprets it:
every value comes from the file's OME metadata (see ``_metadata``), where the
two majors differ, so renamed files remain readable.

Discovery forms stacks of candidate files. ``"processed_single_folder"``
stacks every TIFF below the root, at any depth; ``"processed_multiple_folders"``
makes one stack per immediate subfolder and ignores files in the root::

    run/                       experiment/
    ├── <channel>.tif          ├── <stack_a>/<channel>.tif
    └── <sub>/<channel>.tif    └── <stack_b>/<sub>/<channel>.tiff

- Candidates have a case-insensitive ``.tif`` or ``.tiff`` suffix and no path
  component (below the stack folder) starting with ``.`` or named
  ``__MACOSX``, the folder macOS archives add; hidden folders are not even
  listed, so an unreadable one does no harm. Directories are not candidates;
  every other candidate must be a regular file, so a dangling symlink fails
  instead of silently dropping a channel.
- Candidates are in lexical path order. Whether a candidate holds OME-XML,
  and therefore is a channel, is decided when it is inspected.
- A stack is named after its folder. A folder name that is not a valid
  SpatialData element name is repaired with a warning (``my run`` becomes
  ``my_run``), and folders whose names then agree ignoring case fail before
  any TIFF is read.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, warn_user
from spatialdata_io.readers._utils.names import sanitize_element_name
from spatialdata_io.readers._utils.paths import (
    ArtifactDirectory,
    ArtifactFile,
    DatasetRoot,
    list_directory,
    normalize_dataset_root,
    require_metadata_directory,
    require_metadata_file,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.paths import PathPolicy

type ParsingStyle = Literal["processed_single_folder", "processed_multiple_folders"]

_PARSING_STYLES: tuple[ParsingStyle, ...] = ("processed_single_folder", "processed_multiple_folders")
_CONTEXT = ReaderErrorContext("macsima", "dataset layout")
_NAME_SOURCE = "macsima: the stack folder name"
_TIFF_SUFFIXES = frozenset({".tif", ".tiff"})
# macOS archive metadata; its "._name.tif" entries are not TIFF files.
_ARCHIVE_METADATA = "__MACOSX"


@dataclass(frozen=True, slots=True)
class StackFiles:
    """Candidate channel files of one processed folder, in lexical path order.

    ``name`` is the folder name repaired into a valid SpatialData element
    name; it names the stack's elements and coordinate system. ``directory``
    is the validated folder.
    """

    name: str
    directory: ArtifactDirectory
    files: tuple[ArtifactFile, ...]


@dataclass(frozen=True, slots=True)
class MacsimaLayout:
    """Every stack of one invocation; both parsing styles share this shape."""

    stacks: tuple[StackFiles, ...]


def validate_layout_options(parsing_style: ParsingStyle, exclude_folder_names: Sequence[str]) -> frozenset[str]:
    """Validate the parsing style and own the exact folder names it may exclude."""
    if not isinstance(parsing_style, str):
        message = f"parsing_style must be a string; found {type(parsing_style).__name__}."
        raise TypeError(message)
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
    """Accept a non-directory entry with a TIFF suffix and no hidden path component."""
    return (
        path.suffix.lower() in _TIFF_SUFFIXES
        and all(_visible(part) for part in path.relative_to(directory).parts)
        # A dangling symlink is no directory; validating it below reports it.
        and not path.is_dir()
    )


def _candidate_files(root: DatasetRoot, directory: ArtifactDirectory) -> tuple[ArtifactFile, ...]:
    """Return the candidate TIFFs below a folder of ``root`` in lexical path order."""
    # Hidden paths are never candidates, so hidden folders are not even entered.
    paths = list_directory(directory, context=_CONTEXT, recursive=True, enter_hidden_directories=False)
    return tuple(
        require_metadata_file(root, path, context=_CONTEXT) for path in paths if _is_candidate(path, directory)
    )


def _element_name(folder_name: str) -> str:
    """Repair a folder name into an element name, with a warning if it changes."""
    return sanitize_element_name(folder_name, source=_NAME_SOURCE, parameter=None)


def _require_unique_names(folders: Sequence[tuple[str, StackFiles]], root: DatasetRoot) -> None:
    """Reject folders whose element names agree ignoring case after repair."""
    # SpatialData compares element names with str.lower().
    counts = Counter(stack.name.lower() for _, stack in folders)
    colliding = [folder for folder, stack in folders if counts[stack.name.lower()] > 1]
    if colliding:
        raise ReaderFormatError(
            context=_CONTEXT,
            path=root.path,
            found=f"folders {colliding} whose element names are equal ignoring case",
            expected="folder names that remain distinct as SpatialData element names, ignoring case",
        )


def _single_folder(root: DatasetRoot) -> tuple[StackFiles, ...]:
    files = _candidate_files(root, root.path)
    if not files:
        raise ReaderFormatError(
            context=_CONTEXT, path=root.path, found="no .tif or .tiff files", expected="processed MACSima channel TIFFs"
        )
    return (StackFiles(_element_name(root.path.name), root.path, files),)


def _multiple_folders(root: DatasetRoot, exclude_folder_names: frozenset[str]) -> tuple[StackFiles, ...]:
    entries = [
        entry for entry in list_directory(root.path, context=_CONTEXT) if _visible(entry.name) and entry.is_dir()
    ]
    unknown = exclude_folder_names - {entry.name for entry in entries}
    if unknown:
        message = f"exclude_folder_names {sorted(unknown)} match no folder in {str(root.path)!r}."
        raise ValueError(message)

    # Stacks are named after the folder entry; under the strict policy a
    # symlinked folder's validated directory carries its target's name.
    folders: list[tuple[str, StackFiles]] = []
    for entry in entries:
        if entry.name in exclude_folder_names:
            continue
        directory = require_metadata_directory(root, entry, context=_CONTEXT)
        files = _candidate_files(root, directory)
        if not files:
            warn_user(f"macsima: no TIFF files found in {str(entry)!r}; skipping this folder.")
            continue
        folders.append((entry.name, StackFiles(_element_name(entry.name), directory, files)))

    if not folders:
        raise ReaderFormatError(
            context=_CONTEXT,
            path=root.path,
            found="no included subfolder with .tif or .tiff files",
            expected="at least one processed MACSima folder",
        )
    _require_unique_names(folders, root)
    return tuple(stack for _, stack in folders)


def discover_layout(
    path: str | Path,
    *,
    parsing_style: ParsingStyle,
    exclude_folder_names: frozenset[str],
    path_policy: PathPolicy = "strict",
) -> MacsimaLayout:
    """Resolve the root once and collect the candidate files of every stack.

    Folders named exactly in ``exclude_folder_names`` are omitted, and
    folders without TIFFs are skipped with a warning. Every folder and file
    follows the root's path policy.
    """
    root = normalize_dataset_root(path, context=_CONTEXT, path_policy=path_policy)
    if parsing_style == "processed_single_folder":
        return MacsimaLayout(_single_folder(root))
    return MacsimaLayout(_multiple_folders(root, exclude_folder_names))
