"""Contextual errors and user warnings shared by reader utilities and readers.

:class:`ReaderErrorContext` names the reader, artifact and detected format
version of an operation. :class:`ReaderFormatError` reports a violated reader
contract with that context; :func:`add_reader_context` adds it as a note to an
exception that is re-raised unchanged. :class:`RasterFormatError` is raised by
the shared raster boundary, which has no reader context.

:func:`warn_user` emits a reader's ``UserWarning`` at the user's call site, the
first stack frame outside ``spatialdata_io``, however deeply the reader emits
it.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

# Directory prefix of every module of this package (``.../spatialdata_io/``).
# Warnings skip frames under it, whatever the call depth inside the package.
_PACKAGE_PREFIX = f"{Path(__file__).parents[2]}{os.sep}"


@dataclass(frozen=True, slots=True)
class ReaderErrorContext:
    """Identify the reader artifact and optional detected format version.

    Parameters
    ----------
    reader
        Stable reader name used in diagnostic messages.
    artifact
        Reader-owned semantic name for the artifact being interpreted.
    detected_version
        Optional already-normalized format or software version. Shared
        utilities store this value without interpreting it.

    Notes
    -----
    Values are created by typed reader code rather than parsed from external
    data, so this class does not duplicate static type checks at runtime.
    """

    reader: str
    artifact: str
    detected_version: str | None = None

    def describe(self, *, path: Path | None = None) -> str:
        """Return the context as a stable human-readable description.

        Parameters
        ----------
        path
            Optional artifact or search path associated with the operation.

        Returns
        -------
        str
            Reader, artifact, optional detected version, and optional path.
        """
        description = f"reader {self.reader!r}, artifact {self.artifact!r}"
        if self.detected_version is not None:
            description += f", detected version {self.detected_version!r}"
        if path is not None:
            description += f", path {str(path)!r}"
        return description

    def describe_failure(self, *, found: str, expected: str, path: Path | None = None) -> str:
        """Return the stable message of a violated reader contract.

        Parameters
        ----------
        found
            Concise description of the observed state.
        expected
            Concise description of the required contract.
        path
            Optional artifact or search path associated with the failure.

        Returns
        -------
        str
            ``"<context>: expected <expected>; found <found>."``, shared by
            :class:`ReaderFormatError` and contextual filesystem errors.
        """
        return f"{self.describe(path=path)}: expected {expected}; found {found}."


class ReaderFormatError(ValueError):
    """A reader-declared layout, content, or schema contract was violated.

    Parameters
    ----------
    context
        Reader, artifact, and optional detected-version identity.
    found
        Concise description of the observed state.
    expected
        Concise description of the required contract.
    path
        Optional artifact or search path associated with the failure.

    Notes
    -----
    Structured fields, the rendered message, and exception notes survive
    serialization across process and distributed-execution boundaries.
    """

    context: ReaderErrorContext
    found: str
    expected: str
    path: Path | None

    def __init__(
        self,
        *,
        context: ReaderErrorContext,
        found: str,
        expected: str,
        path: Path | None = None,
    ) -> None:
        """Store the structured fields and render the stable contextual message."""
        self.context = context
        self.found = found
        self.expected = expected
        self.path = path
        super().__init__(context.describe_failure(found=found, expected=expected, path=path))

    def __reduce__(self) -> tuple[object, tuple[ReaderErrorContext, str, str, Path | None], dict[str, object]]:
        """Return structured state for exception transport across processes."""
        return (
            _restore_reader_format_error,
            (self.context, self.found, self.expected, self.path),
            dict(self.__dict__),
        )


class RasterFormatError(ValueError):
    """A raster source violates its declared storage-format contract.

    Raised by the shared raster boundary, which has no reader context, for
    malformed or unsupported encoded content and metadata. Readers add their
    context with :func:`add_reader_context` where they own the artifact.
    """


def add_reader_context(
    error: Exception,
    /,
    *,
    context: ReaderErrorContext,
    path: Path | None = None,
) -> None:
    """Add reader context to an exception that its handler will re-raise.

    Parameters
    ----------
    error
        Exception whose identity, type, traceback, and cause must survive.
    context
        Reader, artifact, and optional detected-version identity.
    path
        Optional artifact or search path associated with the operation.

    Notes
    -----
    Call this function inside a narrow exception handler and then use a bare
    ``raise``. Parser failures converted to :class:`ReaderFormatError` should
    instead use explicit exception chaining.
    """
    error.add_note(f"While handling {context.describe(path=path)}.")


def warn_user(message: str, /) -> None:
    """Emit a ``UserWarning`` attributed to the first caller outside ``spatialdata_io``.

    Parameters
    ----------
    message
        Complete warning text, by convention starting with the reader name
        (``"stereoseq: ..."``). It is emitted unchanged.

    Warns
    -----
    UserWarning
        ``message``, attributed to the first stack frame whose file is not
        inside the ``spatialdata_io`` package: the user's call of a public
        reader, however deeply the reader emits the warning. Warning filters
        that match a module or line, and the once-per-location default,
        apply to that call.

    Notes
    -----
    The frames are skipped with ``skip_file_prefixes`` of
    :func:`warnings.warn` instead of a fixed ``stacklevel``, which would
    point into the package as soon as the emitting call moves to another
    depth.
    """
    warnings.warn(message, UserWarning, skip_file_prefixes=(_PACKAGE_PREFIX,))


def _restore_reader_format_error(
    context: ReaderErrorContext,
    found: str,
    expected: str,
    path: Path | None,
) -> ReaderFormatError:
    """Rebuild a ``ReaderFormatError`` from its pickled structured fields."""
    return ReaderFormatError(context=context, found=found, expected=expected, path=path)


__all__ = ["RasterFormatError", "ReaderErrorContext", "ReaderFormatError", "add_reader_context", "warn_user"]
