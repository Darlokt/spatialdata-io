"""Model-independent reader provenance mutation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata_io import __version__

if TYPE_CHECKING:
    from collections.abc import MutableMapping

_FORMAT_VERSION_KEY = "spatialdata_io_format_version"
_READER_KEY = "spatialdata_io_reader"
_SOFTWARE_VERSION_KEY = "spatialdata_io_software_version"


def set_reader_provenance(
    attrs: MutableMapping[str, object],
    /,
    *,
    reader: str,
    format_version: str | None = None,
) -> None:
    """Record the reader and software versions in an attribute mapping.

    Parameters
    ----------
    attrs
        Borrowed mutable attributes of a newly assembled reader result. The
        mapping is updated in place; unrelated entries are preserved.
    reader
        Already-normalized reader name authored by the calling reader.
    format_version
        Reliably detected and already-normalized vendor format or software
        version. A stale format-version entry is removed when this is ``None``.

    Notes
    -----
    This function performs no artifact access or external-data validation.
    Reader-owned parsing and layout code is responsible for normalizing the
    supplied strings.
    """
    attrs[_SOFTWARE_VERSION_KEY] = __version__
    attrs[_READER_KEY] = reader
    if format_version is None:
        attrs.pop(_FORMAT_VERSION_KEY, None)
    else:
        attrs[_FORMAT_VERSION_KEY] = format_version
