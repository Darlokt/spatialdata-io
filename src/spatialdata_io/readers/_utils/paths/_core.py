"""Deterministic local-filesystem primitives for private reader layouts."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NewType

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context

if TYPE_CHECKING:
    from collections.abc import Iterable

ArtifactDirectory = NewType("ArtifactDirectory", Path)
ArtifactFile = NewType("ArtifactFile", Path)
DatasetRoot = NewType("DatasetRoot", ArtifactDirectory)

type _PathKind = Literal["file", "directory"]
type _PathOrigin = Literal["caller", "metadata"]


def _failure_message(
    context: ReaderErrorContext,
    *,
    found: str,
    expected: str,
    path: Path | None = None,
) -> str:
    return f"{context.describe(path=path)}: expected {expected}; found {found}."


def _lexical_absolute(path: str | Path, *, expand_user: bool) -> Path:
    lexical = Path(path)
    if expand_user:
        lexical = lexical.expanduser()
    return lexical.absolute()


def _missing_error(
    *,
    context: ReaderErrorContext,
    path: Path,
    found: str,
    expected: str,
) -> FileNotFoundError:
    message = _failure_message(context, found=found, expected=expected, path=path)
    return FileNotFoundError(errno.ENOENT, message, str(path))


def _wrong_kind_error(
    *,
    context: ReaderErrorContext,
    path: Path,
    found: str,
    expected: str,
    error_type: type[IsADirectoryError | NotADirectoryError],
) -> IsADirectoryError | NotADirectoryError:
    code = errno.EISDIR if error_type is IsADirectoryError else errno.ENOTDIR
    message = _failure_message(context, found=found, expected=expected, path=path)
    return error_type(code, message, str(path))


def _expected_kind(kind: _PathKind) -> str:
    return "an existing regular file" if kind == "file" else "an existing directory"


def _filesystem_kind(path: Path, *, context: ReaderErrorContext, lexical: Path) -> str:
    try:
        mode = path.stat().st_mode
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

    if stat.S_ISREG(mode):
        return "a regular file"
    if stat.S_ISDIR(mode):
        return "a directory"
    return "a non-regular, non-directory filesystem entry"


def _resolve_existing(path: Path, *, context: ReaderErrorContext, expected: str) -> Path:
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as error:
        raise _missing_error(
            context=context,
            path=path,
            found="a missing path component or dangling symlink",
            expected=expected,
        ) from error
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
    expected = _expected_kind(kind)
    found = _filesystem_kind(target, context=context, lexical=lexical)
    if kind == "file" and found == "a regular file":
        return target
    if kind == "directory" and found == "a directory":
        return target
    if kind == "file" and found == "a directory":
        raise _wrong_kind_error(
            context=context,
            path=lexical,
            found=found,
            expected=expected,
            error_type=IsADirectoryError,
        )
    if kind == "directory" and found == "a regular file":
        raise _wrong_kind_error(
            context=context,
            path=lexical,
            found=found,
            expected=expected,
            error_type=NotADirectoryError,
        )
    raise ReaderFormatError(context=context, found=found, expected=expected, path=lexical)


def _require_kind(path: Path, *, context: ReaderErrorContext, kind: _PathKind) -> Path:
    lexical = _lexical_absolute(path, expand_user=False)
    target = _resolve_existing(lexical, context=context, expected=_expected_kind(kind))
    return _require_resolved_kind(lexical, target, context=context, kind=kind)


def _optional_kind(path: Path, *, context: ReaderErrorContext, kind: _PathKind) -> Path | None:
    lexical = _lexical_absolute(path, expand_user=False)
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

    target = _resolve_existing(lexical, context=context, expected=_expected_kind(kind))
    return _require_resolved_kind(lexical, target, context=context, kind=kind)


def _origin_candidate(root: DatasetRoot, path: str | Path, *, origin: _PathOrigin) -> tuple[Path, bool]:
    supplied = Path(path)
    if origin == "caller":
        supplied = supplied.expanduser()
    is_absolute = supplied.is_absolute()
    candidate = supplied if is_absolute else Path(root) / supplied
    return candidate.absolute(), is_absolute


def _lexically_normalized(path: Path) -> Path:
    """Collapse ``.`` and ``..`` without resolving symlinks or requiring existence."""
    return Path(os.path.normpath(path))


def _require_origin_kind(
    root: DatasetRoot,
    path: str | Path,
    *,
    context: ReaderErrorContext,
    origin: _PathOrigin,
    kind: _PathKind,
) -> Path:
    lexical, is_absolute = _origin_candidate(root, path, origin=origin)
    must_be_contained = origin == "metadata" or not is_absolute
    lexical_target = _lexically_normalized(lexical)
    if must_be_contained and not lexical_target.is_relative_to(root):
        raise ReaderFormatError(
            context=context,
            found=f"a path lexically outside the dataset root at {str(lexical_target)!r}",
            expected=f"a path contained by dataset root {str(root)!r}",
            path=lexical,
        )

    target = _resolve_existing(lexical, context=context, expected=_expected_kind(kind))
    if must_be_contained and not target.is_relative_to(root):
        raise ReaderFormatError(
            context=context,
            found=f"a path resolving outside the dataset root to {str(target)!r}",
            expected=f"a path contained by dataset root {str(root)!r}",
            path=lexical,
        )
    return _require_resolved_kind(lexical, target, context=context, kind=kind)


def _materialize_paths(
    paths: Iterable[Path],
    *,
    context: ReaderErrorContext,
    search_path: Path,
) -> tuple[Path, ...]:
    try:
        return tuple(paths)
    except OSError as error:
        add_reader_context(error, context=context, path=search_path)
        raise


def _format_candidates(paths: tuple[Path, ...]) -> str:
    rendered = ", ".join(repr(str(path)) for path in paths)
    return f"{len(paths)} candidates ({rendered})"


def normalize_dataset_root(
    path: str | Path,
    /,
    *,
    context: ReaderErrorContext,
) -> DatasetRoot:
    """Return a canonical existing dataset directory.

    Parameters
    ----------
    path
        Local dataset-root path. User expansion and strict symlink resolution
        are applied before validating the directory kind.
    context
        Reader context included in failures.

    Returns
    -------
    DatasetRoot
        Canonical absolute directory path. This allocation-free ``NewType`` is
        also an :class:`ArtifactDirectory` and records successful root
        normalization for downstream layout functions.

    Raises
    ------
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
    lexical = _lexical_absolute(path, expand_user=True)
    target = _resolve_existing(lexical, context=context, expected="an existing directory")
    directory = ArtifactDirectory(_require_resolved_kind(lexical, target, context=context, kind="directory"))
    return DatasetRoot(directory)


def require_file(path: Path, /, *, context: ReaderErrorContext) -> ArtifactFile:
    """Return the canonical target of a required regular file.

    Parameters
    ----------
    path
        Reader-owned local artifact path. User expansion is not applied.
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


def optional_file(path: Path, /, *, context: ReaderErrorContext) -> ArtifactFile | None:
    """Return a canonical regular file, or ``None`` for an absent leaf.

    Parameters
    ----------
    path
        Reader-owned local artifact path. User expansion is not applied. Only
        the final path component is optional; its parent must be a valid
        existing directory.
    context
        Reader context included in failures.

    Returns
    -------
    ArtifactFile | None
        Canonical absolute file target, or ``None`` when only the final path
        component is absent.

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
        If an existing path resolves to another unsupported filesystem kind.
    OSError
        If another filesystem operation fails.
    """
    result = _optional_kind(path, context=context, kind="file")
    return None if result is None else ArtifactFile(result)


def require_directory(path: Path, /, *, context: ReaderErrorContext) -> ArtifactDirectory:
    """Return the canonical target of a required directory.

    Parameters
    ----------
    path
        Reader-owned local directory path. User expansion is not applied.
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


def optional_directory(path: Path, /, *, context: ReaderErrorContext) -> ArtifactDirectory | None:
    """Return a canonical directory, or ``None`` for an absent leaf.

    Parameters
    ----------
    path
        Reader-owned local directory path. User expansion is not applied. Only
        the final path component is optional; its parent must be a valid
        existing directory.
    context
        Reader context included in failures.

    Returns
    -------
    ArtifactDirectory | None
        Canonical absolute directory target, or ``None`` when only the final
        path component is absent. Directory contents are not scanned.

    Raises
    ------
    FileNotFoundError
        If a parent is missing or dangling, or an existing leaf is a dangling
        symlink.
    NotADirectoryError
        If the leaf or its parent resolves to a regular file.
    ReaderFormatError
        If an existing path resolves to another unsupported filesystem kind.
    OSError
        If another filesystem operation fails.
    """
    result = _optional_kind(path, context=context, kind="directory")
    return None if result is None else ArtifactDirectory(result)


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
        Canonical dataset root returned by :func:`normalize_dataset_root`.
    path
        Caller-provided path. User expansion is applied. Relative paths resolve
        against and must remain within ``root``; explicit absolute paths may
        resolve outside it.
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
        If a relative path escapes ``root`` or resolves to an unsupported kind.
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
        Canonical dataset root returned by :func:`normalize_dataset_root`.
    path
        Caller-provided path. User expansion is applied. Relative paths resolve
        against and must remain within ``root``; explicit absolute paths may
        resolve outside it.
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
        If a relative path escapes ``root`` or resolves to an unsupported kind.
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
    """Resolve a literal metadata-derived file path contained by ``root``.

    Parameters
    ----------
    root
        Canonical dataset root returned by :func:`normalize_dataset_root`.
    path
        Path obtained from vendor or community metadata. User expansion is not
        applied. Relative and absolute values must resolve within ``root``.
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
        If the target escapes ``root`` or has an unsupported filesystem kind.
    OSError
        If another filesystem operation fails.
    """
    return ArtifactFile(_require_origin_kind(root, path, context=context, origin="metadata", kind="file"))


def require_metadata_directory(
    root: DatasetRoot,
    path: str | Path,
    /,
    *,
    context: ReaderErrorContext,
) -> ArtifactDirectory:
    """Resolve a literal metadata-derived directory path contained by ``root``.

    Parameters
    ----------
    root
        Canonical dataset root returned by :func:`normalize_dataset_root`.
    path
        Path obtained from vendor or community metadata. User expansion is not
        applied. Relative and absolute values must resolve within ``root``.
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
        If the target escapes ``root`` or has an unsupported filesystem kind.
    OSError
        If another filesystem operation fails.
    """
    return ArtifactDirectory(_require_origin_kind(root, path, context=context, origin="metadata", kind="directory"))


def sorted_paths(
    paths: Iterable[Path],
    /,
    *,
    context: ReaderErrorContext,
    search_path: Path,
) -> tuple[Path, ...]:
    """Materialize one path iterable in deterministic lexical order.

    Parameters
    ----------
    paths
        Reader-discovered paths. The iterable is consumed exactly once.
    context
        Reader context added to filesystem iteration failures.
    search_path
        Directory or tree root from which the reader obtained ``paths``. It is
        diagnostic context only and is not accessed or validated here.

    Returns
    -------
    tuple[Path, ...]
        Immutable paths sorted by POSIX lexical spelling. Paths are not
        resolved, deduplicated, accessed, or filtered by kind.

    Raises
    ------
    OSError
        If materializing a filesystem-backed iterable fails. The original
        exception is retained with reader and search-path context.
    """
    try:
        return tuple(sorted(paths, key=Path.as_posix))
    except OSError as error:
        add_reader_context(error, context=context, path=search_path)
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
    candidates = _materialize_paths(paths, context=context, search_path=search_path)
    if not candidates:
        raise _missing_error(
            context=context,
            path=search_path,
            found="no candidates",
            expected="exactly one candidate",
        )
    if len(candidates) > 1:
        ordered = tuple(sorted(candidates, key=Path.as_posix))
        raise ReaderFormatError(
            context=context,
            found=_format_candidates(ordered),
            expected="exactly one candidate",
            path=search_path,
        )
    return candidates[0]


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
    candidates = _materialize_paths(paths, context=context, search_path=search_path)
    if not candidates:
        return None
    if len(candidates) > 1:
        ordered = tuple(sorted(candidates, key=Path.as_posix))
        raise ReaderFormatError(
            context=context,
            found=_format_candidates(ordered),
            expected="zero or one candidate",
            path=search_path,
        )
    return candidates[0]


__all__ = [
    "ArtifactDirectory",
    "ArtifactFile",
    "DatasetRoot",
    "normalize_dataset_root",
    "optional_directory",
    "optional_file",
    "optional_unique_path",
    "require_caller_directory",
    "require_caller_file",
    "require_directory",
    "require_file",
    "require_metadata_directory",
    "require_metadata_file",
    "require_unique_path",
    "sorted_paths",
]
