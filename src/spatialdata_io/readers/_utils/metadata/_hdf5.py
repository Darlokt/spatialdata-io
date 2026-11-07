r"""Decode one HDF5 attribute value as strict, nonempty UTF-8 text.

10x and other producers store text attributes either as fixed-length byte
strings, which h5py returns as ``numpy.bytes_``, or as variable-length strings,
which h5py returns as ``str``. h5py decodes variable-length strings with the
``surrogateescape`` error handler, so invalid UTF-8 arrives as a ``str``
holding lone surrogates (``b"\xff"`` becomes ``"\udcff"``)::

    stored                     h5py returns         result
    b"spaceranger-2.1.0" (S)   np.bytes_(...)       "spaceranger-2.1.0"
    "spaceranger-2.1.0" (vlen) "spaceranger-2.1.0"  "spaceranger-2.1.0"
    b"\xff" (vlen)             "\udcff"             ReaderFormatError
    0-d string array           np.ndarray, ndim 0   its element, as above

The decoder reverses the ``surrogateescape`` step and decodes strictly, so the
two storage forms obey one contract. It does not open files: readers read the
attribute while they own the HDF5 store and pass the value in. Arrays with
elements (such as ``library_ids``) and the meaning of the text stay with the
reader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    from pathlib import Path

    from spatialdata_io.readers._utils.errors import ReaderErrorContext


def decode_hdf5_text(value: object, /, *, attribute: str, context: ReaderErrorContext, path: Path) -> str:
    """Return an HDF5 attribute value as strict, nonempty UTF-8 text.

    Parameters
    ----------
    value
        Attribute value as returned by h5py: ``bytes``, ``numpy.bytes_``,
        ``str``, ``numpy.str_``, or a 0-d array of one of these.
    attribute
        Attribute name, used in error messages.
    context
        Owning reader and artifact for error messages.
    path
        HDF5 file that stores the attribute.

    Returns
    -------
    str
        The decoded text.

    Raises
    ------
    ReaderFormatError
        If the value is not text, is not valid UTF-8 (including lone
        surrogates from h5py's ``surrogateescape`` decoding), or is empty.
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, str | np.str_):
        # Recover the stored bytes, so that invalid UTF-8 is not accepted as text.
        try:
            value = str(value).encode("utf-8", errors="surrogateescape")
        except UnicodeEncodeError as error:
            raise _not_utf8(attribute, context=context, path=path) from error
    if not isinstance(value, bytes | np.bytes_):
        raise ReaderFormatError(
            context=context,
            found=f"attribute {attribute!r} with value type {type(value).__name__}",
            expected="UTF-8 text",
            path=path,
        )

    try:
        text = bytes(value).decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise _not_utf8(attribute, context=context, path=path) from error
    if not text:
        raise ReaderFormatError(
            context=context, found=f"empty attribute {attribute!r}", expected="nonempty text", path=path
        )
    return text


def _not_utf8(attribute: str, *, context: ReaderErrorContext, path: Path) -> ReaderFormatError:
    """Describe an attribute whose stored bytes are not UTF-8."""
    return ReaderFormatError(
        context=context, found=f"non-UTF-8 bytes in attribute {attribute!r}", expected="UTF-8 text", path=path
    )
