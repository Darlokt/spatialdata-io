"""Contextual errors shared by private reader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


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
        self.context = context
        self.found = found
        self.expected = expected
        self.path = path
        super().__init__(f"{context.describe(path=path)}: expected {expected}; found {found}.")

    def __reduce__(self) -> tuple[object, tuple[ReaderErrorContext, str, str, Path | None], dict[str, object]]:
        """Return structured state for exception transport across processes."""
        return (
            _restore_reader_format_error,
            (self.context, self.found, self.expected, self.path),
            dict(self.__dict__),
        )


class RasterFormatError(ValueError):
    """A raster source violates its declared storage-format contract.

    This specialized type distinguishes malformed encoded raster content and
    metadata from reader-layout and caller-option failures. It intentionally
    remains a ``ValueError`` for compatibility with existing raster callers.
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


def _restore_reader_format_error(
    context: ReaderErrorContext,
    found: str,
    expected: str,
    path: Path | None,
) -> ReaderFormatError:
    return ReaderFormatError(context=context, found=found, expected=expected, path=path)


__all__ = ["RasterFormatError", "ReaderErrorContext", "ReaderFormatError", "add_reader_context"]
