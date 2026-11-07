"""Deterministic local-filesystem primitives for private reader layouts.

Every artifact that a reader resolves from, or discovers under, a dataset root
obeys the root's path policy, chosen once per reader call:

- ``"strict"``: the artifact must resolve, symlinks followed, to a target
  inside the dataset root. Otherwise :class:`ReaderFormatError` is raised.
- ``"relaxed"``: the artifact may be a symlink whose target lies anywhere, as in
  git-annex and DVC datasets.

Paths have three origins:

- Dataset paths (a reader-owned name, a discovered entry, or a value read from
  metadata) must be lexically inside the root in both modes: an absolute path
  elsewhere or a ``..`` escape is rejected before any filesystem access. Their
  targets then follow the policy. A strict root returns the canonical target.
- Relative caller paths are joined to the root and follow the same rules,
  except that ``~`` is expanded first.
- Absolute caller paths may point anywhere under both policies.

Caller paths keep the final name the caller gave under both policies, and
every path of a relaxed root keeps its own final name: the validator returns
the resolved parent directory joined with that name. Names and suffixes that
identify samples or formats therefore come from the dataset or the caller,
never from a hash-named target.
"""

from __future__ import annotations

import errno
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NewType

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

ArtifactDirectory = NewType("ArtifactDirectory", Path)
ArtifactFile = NewType("ArtifactFile", Path)

type PathPolicy = Literal["strict", "relaxed"]
type _PathKind = Literal["file", "directory"]
type _PathOrigin = Literal["caller", "metadata"]

# macOS writes "._<name>" AppleDouble metadata companions on foreign volumes.
_APPLE_DOUBLE_PREFIX = "._"
_EXPECTED_KIND: dict[_PathKind, str] = {"file": "an existing regular file", "directory": "an existing directory"}


@dataclass(frozen=True, slots=True)
class DatasetRoot:
    """A canonical dataset directory and the path policy of its artifacts.

    Only :func:`normalize_dataset_root` constructs this value.

    Attributes
    ----------
    path
        Canonical absolute dataset directory.
    path_policy
        ``"strict"`` if every dataset artifact must resolve inside ``path``;
        ``"relaxed"`` if dataset artifacts may be symlinks to targets elsewhere.
    """

    path: ArtifactDirectory
    path_policy: PathPolicy


def _missing_error(
    *,
    context: ReaderErrorContext,
    path: Path,
    found: str,
    expected: str,
) -> FileNotFoundError:
    """Build a ``FileNotFoundError`` with ``ENOENT``, ``filename`` and the contextual message."""
    message = context.describe_failure(found=found, expected=expected, path=path)
    return FileNotFoundError(errno.ENOENT, message, str(path))


def _resolve_existing(path: Path, *, context: ReaderErrorContext, expected: str) -> Path:
    """Resolve ``path`` strictly, reporting a missing component as a contextual error."""
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as error:
        raise _missing_error(
            context=context,
            path=path,
            found="a missing path component or dangling symlink",
            expected=expected,
        ) from error
    # Python 3.12 reports symlink loops as RuntimeError, later versions as OSError.
    except (OSError, RuntimeError) as error:
        add_reader_context(error, context=context, path=path)
        raise


def _require_resolved_kind(
    lexical: Path,
    target: Path,
    *,
    context: ReaderErrorContext,
    kind: _PathKind,
) -> Path:
    """Return ``target`` if it has ``kind``, reporting failures at ``lexical``."""
    try:
        mode = target.stat().st_mode
    except FileNotFoundError as error:
        raise _missing_error(
            context=context,
            path=lexical,
            found="a target that disappeared during validation",
            expected="an existing filesystem entry",
        ) from error
    except OSError as error:
        add_reader_context(error, context=context, path=lexical)
        raise

    if (kind == "file" and stat.S_ISREG(mode)) or (kind == "directory" and stat.S_ISDIR(mode)):
        return target
    expected = _EXPECTED_KIND[kind]
    if kind == "file" and stat.S_ISDIR(mode):
        message = context.describe_failure(found="a directory", expected=expected, path=lexical)
        raise IsADirectoryError(errno.EISDIR, message, str(lexical))
    if kind == "directory" and stat.S_ISREG(mode):
        message = context.describe_failure(found="a regular file", expected=expected, path=lexical)
        raise NotADirectoryError(errno.ENOTDIR, message, str(lexical))
    raise ReaderFormatError(
        context=context,
        found="a non-regular, non-directory filesystem entry",
        expected=expected,
        path=lexical,
    )


def _require_kind(path: Path, *, context: ReaderErrorContext, kind: _PathKind) -> Path:
    """Return the resolved target of ``path`` after checking that it has ``kind``."""
    lexical = path.absolute()
    target = _resolve_existing(lexical, context=context, expected=_EXPECTED_KIND[kind])
    return _require_resolved_kind(lexical, target, context=context, kind=kind)


def _validate_path_policy(value: object, *, context: ReaderErrorContext) -> PathPolicy:
    """Return a public ``path_policy`` value after checking its type and spelling."""
    if not isinstance(value, str):
        message = f"{context.reader}: path_policy must be a string, not {type(value).__name__}."
        raise TypeError(message)
    if value == "strict":
        return "strict"
    if value == "relaxed":
        return "relaxed"
    message = f"{context.reader}: path_policy must be 'strict' or 'relaxed', not {value!r}."
    raise ValueError(message)


def _contained_lexical(root: DatasetRoot, supplied: Path, *, context: ReaderErrorContext) -> Path:
    """Join ``supplied`` to ``root`` and reject a spelling that leaves it, without filesystem access."""
    lexical = (root.path / supplied).absolute()
    # Collapse ``.`` and ``..`` without resolving symlinks or requiring existence.
    normalized = Path(os.path.normpath(lexical))
    if not normalized.is_relative_to(root.path):
        raise ReaderFormatError(
            context=context,
            found=f"a path lexically outside the dataset root at {str(normalized)!r}",
            expected=f"a path contained by dataset root {str(root.path)!r}",
            path=lexical,
        )
    return lexical


def _require_origin_kind(
    root: DatasetRoot,
    path: str | Path,
    *,
    context: ReaderErrorContext,
    origin: _PathOrigin,
    kind: _PathKind,
) -> Path:
    """Resolve a caller or metadata path against ``root`` and require it to have ``kind``.

    Metadata paths and relative caller paths must be lexically inside ``root``;
    under a strict root their targets must also resolve inside it. Absolute
    caller paths are permitted anywhere. Caller paths, and every path of a
    relaxed root, keep their own final name; strict metadata paths return the
    canonical target.
    """
    supplied = Path(path).expanduser() if origin == "caller" else Path(path)
    must_be_contained = origin == "metadata" or not supplied.is_absolute()
    lexical = (
        _contained_lexical(root, supplied, context=context) if must_be_contained else (root.path / supplied).absolute()
    )

    target = _resolve_existing(lexical, context=context, expected=_EXPECTED_KIND[kind])
    relaxed = root.path_policy == "relaxed"
    if must_be_contained and not relaxed and not target.is_relative_to(root.path):
        raise ReaderFormatError(
            context=context,
            found=f"a path resolving outside the dataset root to {str(target)!r}",
            expected=(
                f"a path contained by dataset root {str(root.path)!r} "
                "(pass path_policy='relaxed' to accept symlinks to targets stored elsewhere)"
            ),
            path=lexical,
        )
    _require_resolved_kind(lexical, target, context=context, kind=kind)
    if origin == "metadata" and not relaxed:
        return target
    # Annex and DVC targets are named by content hashes: keep the given name and suffix.
    parent = _resolve_existing(lexical.parent, context=context, expected="an existing parent directory")
    return parent / lexical.name


def _optional_metadata_kind(
    root: DatasetRoot,
    path: str | Path,
    *,
    context: ReaderErrorContext,
    kind: _PathKind,
) -> Path | None:
    """Return a validated metadata path, or ``None`` when only its final component is absent.

    The parent directory must exist. A dangling symlink is not absent and raises.
    """
    lexical = _contained_lexical(root, Path(path), context=context)
    try:
        lexical.lstat()
    except FileNotFoundError:
        parent = lexical.parent
        parent_target = _resolve_existing(
            parent,
            context=context,
            expected="an existing parent directory for an optional artifact",
        )
        _require_resolved_kind(parent, parent_target, context=context, kind="directory")
        return None
    except OSError as error:
        add_reader_context(error, context=context, path=lexical)
        raise
    return _require_origin_kind(root, lexical, context=context, origin="metadata", kind=kind)


def _unique_candidate(
    paths: Iterable[Path],
    *,
    context: ReaderErrorContext,
    search_path: Path,
    expected: str,
) -> Path | None:
    """Return the only candidate, ``None`` when there is none, and reject several."""
    try:
        candidates = tuple(paths)
    except OSError as error:
        add_reader_context(error, context=context, path=search_path)
        raise
    if len(candidates) > 1:
        rendered = ", ".join(repr(str(path)) for path in sorted(candidates, key=Path.as_posix))
        raise ReaderFormatError(
            context=context,
            found=f"{len(candidates)} candidates ({rendered})",
            expected=expected,
            path=search_path,
        )
    return candidates[0] if candidates else None


def normalize_dataset_root(
    path: str | Path,
    /,
    *,
    context: ReaderErrorContext,
    path_policy: PathPolicy,
) -> DatasetRoot:
    """Return a canonical existing dataset directory and its path policy.

    Parameters
    ----------
    path
        Local dataset-root path. User expansion and strict symlink resolution
        are applied before validating the directory kind.
    context
        Reader context included in failures.
    path_policy
        Public reader option deciding whether dataset artifacts must resolve
        inside the root (``"strict"``) or may be symlinks to targets elsewhere
        (``"relaxed"``). It is validated before any filesystem access.

    Returns
    -------
    DatasetRoot
        Canonical absolute directory with the validated policy. Every
        root-relative validator below reads the policy from this value.

    Raises
    ------
    TypeError
        If ``path_policy`` is not a string.
    ValueError
        If ``path_policy`` is neither ``"strict"`` nor ``"relaxed"``.
    FileNotFoundError
        If the root or a symlink target is missing.
    NotADirectoryError
        If the root resolves to a regular file.
    ReaderFormatError
        If the root resolves to another unsupported filesystem kind.
    OSError
        If another filesystem operation fails. The original exception is
        retained with reader context.
    """
    policy = _validate_path_policy(path_policy, context=context)
    directory = _require_kind(Path(path).expanduser(), context=context, kind="directory")
    return DatasetRoot(ArtifactDirectory(directory), policy)


def require_file(path: Path, /, *, context: ReaderErrorContext) -> ArtifactFile:
    """Return the canonical target of a required regular file outside any dataset root.

    Use it for a single caller-selected file without a dataset root or for a
    file inside a reader-created directory. Artifacts under a dataset root use
    the root-relative validators, which apply the root's path policy.

    Parameters
    ----------
    path
        Local file path. User expansion is not applied.
    context
        Reader context included in failures.

    Returns
    -------
    ArtifactFile
        Canonical absolute path to the validated regular file.

    Raises
    ------
    FileNotFoundError
        If the path or a symlink target is missing.
    IsADirectoryError
        If the path resolves to a directory.
    ReaderFormatError
        If the path resolves to another unsupported filesystem kind.
    OSError
        If another filesystem operation fails.
    """
    return ArtifactFile(_require_kind(path, context=context, kind="file"))


def require_directory(path: Path, /, *, context: ReaderErrorContext) -> ArtifactDirectory:
    """Return the canonical target of a required directory outside any dataset root.

    Use it for a reader-created directory such as an extraction target.
    Directories under a dataset root use the root-relative validators.

    Parameters
    ----------
    path
        Local directory path. User expansion is not applied.
    context
        Reader context included in failures.

    Returns
    -------
    ArtifactDirectory
        Canonical absolute path to the validated directory. Its contents are
        not scanned.

    Raises
    ------
    FileNotFoundError
        If the path or a symlink target is missing.
    NotADirectoryError
        If the path resolves to a regular file.
    ReaderFormatError
        If the path resolves to another unsupported filesystem kind.
    OSError
        If another filesystem operation fails.
    """
    return ArtifactDirectory(_require_kind(path, context=context, kind="directory"))


def require_caller_file(
    root: DatasetRoot,
    path: str | Path,
    /,
    *,
    context: ReaderErrorContext,
) -> ArtifactFile:
    """Resolve and validate a caller-provided regular file.

    Parameters
    ----------
    root
        Dataset root returned by :func:`normalize_dataset_root`.
    path
        Caller-provided path. User expansion is applied. Relative paths join
        ``root``, must be lexically inside it, and follow its path policy: a
        strict root requires the target to resolve inside it. Absolute paths
        may point anywhere under both policies.
    context
        Reader context included in failures.

    Returns
    -------
    ArtifactFile
        The resolved parent directory joined with the final name the caller
        gave, so its name and suffix are the caller's, not a link target's.

    Raises
    ------
    FileNotFoundError
        If the path or a symlink target is missing.
    IsADirectoryError
        If the path resolves to a directory.
    ReaderFormatError
        If a relative path is lexically outside ``root``, a strict root's
        relative path resolves outside it, or the target has an unsupported
        filesystem kind.
    OSError
        If another filesystem operation fails.
    """
    return ArtifactFile(_require_origin_kind(root, path, context=context, origin="caller", kind="file"))


def require_caller_directory(
    root: DatasetRoot,
    path: str | Path,
    /,
    *,
    context: ReaderErrorContext,
) -> ArtifactDirectory:
    """Resolve and validate a caller-provided directory.

    Parameters
    ----------
    root
        Dataset root returned by :func:`normalize_dataset_root`.
    path
        Caller-provided path. User expansion is applied. Relative paths join
        ``root``, must be lexically inside it, and follow its path policy: a
        strict root requires the target to resolve inside it. Absolute paths
        may point anywhere under both policies.
    context
        Reader context included in failures.

    Returns
    -------
    ArtifactDirectory
        The resolved parent directory joined with the final name the caller
        gave, so its name and suffix are the caller's, not a link target's. Its contents are
        not scanned.

    Raises
    ------
    FileNotFoundError
        If the path or a symlink target is missing.
    NotADirectoryError
        If the path resolves to a regular file.
    ReaderFormatError
        If a relative path is lexically outside ``root``, a strict root's
        relative path resolves outside it, or the target has an unsupported
        filesystem kind.
    OSError
        If another filesystem operation fails.
    """
    return ArtifactDirectory(_require_origin_kind(root, path, context=context, origin="caller", kind="directory"))


def require_metadata_file(
    root: DatasetRoot,
    path: str | Path,
    /,
    *,
    context: ReaderErrorContext,
) -> ArtifactFile:
    """Resolve a dataset-determined file path under ``root`` with the root's path policy.

    Parameters
    ----------
    root
        Dataset root returned by :func:`normalize_dataset_root`.
    path
        Reader-owned name, discovered entry, or literal value read from
        metadata. User expansion is not applied. Relative values join
        ``root``; relative and absolute values must be lexically inside it.
    context
        Reader context included in failures.

    Returns
    -------
    ArtifactFile
        For a strict root, the canonical target inside ``root``. For a relaxed
        root, the canonical parent directory joined with the artifact's own
        name, whose symlink target may lie anywhere.

    Raises
    ------
    FileNotFoundError
        If the path or a symlink target is missing.
    IsADirectoryError
        If the path resolves to a directory.
    ReaderFormatError
        If the path is lexically outside ``root``, a strict root's artifact
        resolves outside it, or the target has an unsupported filesystem kind.
    OSError
        If another filesystem operation fails.
    """
    return ArtifactFile(_require_origin_kind(root, path, context=context, origin="metadata", kind="file"))


def optional_metadata_file(
    root: DatasetRoot,
    path: str | Path,
    /,
    *,
    context: ReaderErrorContext,
) -> ArtifactFile | None:
    """Resolve an optional dataset-determined file, or return ``None`` for an absent leaf.

    Parameters
    ----------
    root
        Dataset root returned by :func:`normalize_dataset_root`.
    path
        Path accepted by :func:`require_metadata_file`. Only the final path
        component is optional; its parent must be an existing directory.
    context
        Reader context included in failures.

    Returns
    -------
    ArtifactFile | None
        The value :func:`require_metadata_file` returns, or ``None`` when only
        the final path component is absent.

    Raises
    ------
    FileNotFoundError
        If a parent is missing or dangling, or an existing leaf is a dangling
        symlink.
    NotADirectoryError
        If the leaf's parent is not a directory.
    IsADirectoryError
        If the leaf resolves to a directory.
    ReaderFormatError
        Under the conditions of :func:`require_metadata_file`.
    OSError
        If another filesystem operation fails.
    """
    result = _optional_metadata_kind(root, path, context=context, kind="file")
    return None if result is None else ArtifactFile(result)


def require_metadata_directory(
    root: DatasetRoot,
    path: str | Path,
    /,
    *,
    context: ReaderErrorContext,
) -> ArtifactDirectory:
    """Resolve a dataset-determined directory path under ``root`` with the root's path policy.

    Parameters
    ----------
    root
        Dataset root returned by :func:`normalize_dataset_root`.
    path
        Reader-owned name, discovered entry, or literal value read from
        metadata. User expansion is not applied. Relative values join
        ``root``; relative and absolute values must be lexically inside it.
    context
        Reader context included in failures.

    Returns
    -------
    ArtifactDirectory
        For a strict root, the canonical target inside ``root``. For a relaxed
        root, the canonical parent directory joined with the directory's own
        name. Paths joined to the result stay lexically inside ``root`` in both
        modes. Its contents are not scanned.

    Raises
    ------
    FileNotFoundError
        If the path or a symlink target is missing.
    NotADirectoryError
        If the path resolves to a regular file.
    ReaderFormatError
        If the path is lexically outside ``root``, a strict root's artifact
        resolves outside it, or the target has an unsupported filesystem kind.
    OSError
        If another filesystem operation fails.
    """
    return ArtifactDirectory(_require_origin_kind(root, path, context=context, origin="metadata", kind="directory"))


def optional_metadata_directory(
    root: DatasetRoot,
    path: str | Path,
    /,
    *,
    context: ReaderErrorContext,
) -> ArtifactDirectory | None:
    """Resolve an optional dataset-determined directory, or return ``None`` for an absent leaf.

    Parameters
    ----------
    root
        Dataset root returned by :func:`normalize_dataset_root`.
    path
        Path accepted by :func:`require_metadata_directory`. Only the final
        path component is optional; its parent must be an existing directory.
    context
        Reader context included in failures.

    Returns
    -------
    ArtifactDirectory | None
        The value :func:`require_metadata_directory` returns, or ``None`` when
        only the final path component is absent. Contents are not scanned.

    Raises
    ------
    FileNotFoundError
        If a parent is missing or dangling, or an existing leaf is a dangling
        symlink.
    NotADirectoryError
        If the leaf or its parent resolves to a regular file.
    ReaderFormatError
        Under the conditions of :func:`require_metadata_directory`.
    OSError
        If another filesystem operation fails.
    """
    result = _optional_metadata_kind(root, path, context=context, kind="directory")
    return None if result is None else ArtifactDirectory(result)


def _raise(error: OSError) -> None:
    """Let :func:`os.walk` propagate a listing failure instead of skipping the directory."""
    raise error


def _is_apple_double(path: Path) -> bool:
    """Return whether ``path`` names a macOS AppleDouble companion (``._<name>``)."""
    return path.name.startswith(_APPLE_DOUBLE_PREFIX)


def _walk(directory: Path) -> Iterator[Path]:
    """Yield every entry below ``directory``, without following directory symlinks."""
    for parent, directories, files in os.walk(directory, onerror=_raise):
        # Pruning in place stops the walk from descending into AppleDouble entries.
        directories[:] = [name for name in directories if not name.startswith(_APPLE_DOUBLE_PREFIX)]
        base = Path(parent)
        yield from (base / name for name in directories)
        yield from (base / name for name in files)


def list_directory(directory: Path, /, *, context: ReaderErrorContext, recursive: bool = False) -> tuple[Path, ...]:
    """List the entries of one directory in deterministic lexical order.

    macOS AppleDouble companions, entries whose name starts with ``._``, are
    left out: macOS writes one beside every file it copies to a volume
    without extended attributes, and they hold metadata, never data. Other
    hidden entries are listed.

    Parameters
    ----------
    directory
        Directory to list. Readers pass a validated artifact directory.
    context
        Reader context added to listing failures.
    recursive
        Whether to list the entries of every subdirectory too. Symlinked
        subdirectories are listed but not entered, and AppleDouble
        directories are not entered.

    Returns
    -------
    tuple[Path, ...]
        Entry paths (``directory / name``), files and directories alike,
        sorted by POSIX lexical spelling. They are not resolved, accessed or
        filtered by kind.

    Raises
    ------
    OSError
        If ``directory`` or, with ``recursive``, a subdirectory cannot be
        listed. The original exception is retained with reader and directory
        context; no directory is skipped silently.
    """
    try:
        entries = _walk(directory) if recursive else directory.iterdir()
        return tuple(sorted((path for path in entries if not _is_apple_double(path)), key=Path.as_posix))
    except OSError as error:
        add_reader_context(error, context=context, path=directory)
        raise


def require_unique_path(
    paths: Iterable[Path],
    /,
    *,
    context: ReaderErrorContext,
    search_path: Path,
) -> Path:
    """Return exactly one reader-filtered candidate.

    Parameters
    ----------
    paths
        Reader-filtered candidates. The iterable is consumed exactly once.
        Candidates may be files or directories; selection is kind-agnostic.
    context
        Reader context included in failures.
    search_path
        Directory or tree root that was searched. It is diagnostic context
        only and is not accessed or validated here.

    Returns
    -------
    Path
        Sole candidate without resolution or kind validation. Successful
        selection does not sort.

    Raises
    ------
    FileNotFoundError
        If there are no candidates.
    ReaderFormatError
        If there are multiple candidates. All conflicts are reported in
        deterministic lexical order.
    OSError
        If materializing a filesystem-backed iterable fails. The original
        exception is retained with reader and search-path context.
    """
    candidate = _unique_candidate(paths, context=context, search_path=search_path, expected="exactly one candidate")
    if candidate is None:
        raise _missing_error(
            context=context,
            path=search_path,
            found="no candidates",
            expected="exactly one candidate",
        )
    return candidate


def optional_unique_path(
    paths: Iterable[Path],
    /,
    *,
    context: ReaderErrorContext,
    search_path: Path,
) -> Path | None:
    """Return zero or one reader-filtered candidate.

    Parameters
    ----------
    paths
        Reader-filtered candidates. The iterable is consumed exactly once.
        Candidates may be files or directories; selection is kind-agnostic.
    context
        Reader context included in failures.
    search_path
        Directory or tree root that was searched. It is diagnostic context
        only and is not accessed or validated here.

    Returns
    -------
    Path | None
        ``None`` for no candidates or the sole candidate without resolution or
        kind validation. Successful selection does not sort.

    Raises
    ------
    ReaderFormatError
        If there are multiple candidates. All conflicts are reported in
        deterministic lexical order.
    OSError
        If materializing a filesystem-backed iterable fails. The original
        exception is retained with reader and search-path context.
    """
    return _unique_candidate(paths, context=context, search_path=search_path, expected="zero or one candidate")


__all__ = [
    "ArtifactDirectory",
    "ArtifactFile",
    "DatasetRoot",
    "PathPolicy",
    "list_directory",
    "normalize_dataset_root",
    "optional_metadata_directory",
    "optional_metadata_file",
    "optional_unique_path",
    "require_caller_directory",
    "require_caller_file",
    "require_directory",
    "require_file",
    "require_metadata_directory",
    "require_metadata_file",
    "require_unique_path",
]
