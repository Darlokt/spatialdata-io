"""Decode Xenium ``(prefix, suffix)`` uint32 cell-ID pairs to their documented strings.

10x stores cell IDs in ``cells.zarr.zip`` as uint32 pairs and as text everywhere
else. The documented codec writes the prefix as eight hexadecimal digits, maps
each digit 0-f to a-p, and appends ``-`` and the decimal suffix: prefix
1437536272 (``55af1010``) with suffix 1 becomes ``ffkpbaba-1``.
https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/advanced/xoa-output-zarr#cellID
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_NIBBLE_SHIFTS = np.arange(28, -1, -4, dtype=np.uint32)
_PAIR_SHAPE = 2


def decode_cell_ids(pairs: NDArray[np.uint32]) -> NDArray[np.str_]:
    """Return the cell-ID strings of an ``(N, 2)`` uint32 prefix/suffix array.

    Raises
    ------
    ValueError
        If ``pairs`` is not an ``(N, 2)`` uint32 array.
    """
    if pairs.dtype != np.uint32 or pairs.ndim != _PAIR_SHAPE or pairs.shape[1] != _PAIR_SHAPE:
        message = f"Cell-ID pairs must be an (N, 2) uint32 array; found shape {pairs.shape} and dtype {pairs.dtype}."
        raise ValueError(message)

    # Most significant nibble first; each nibble 0-15 becomes the byte "a"-"p".
    nibbles = (pairs[:, :1] >> _NIBBLE_SHIFTS) & 0xF
    prefixes = (nibbles + ord("a")).astype(np.uint8).view("S8").ravel().astype(np.str_)
    return np.strings.add(np.strings.add(prefixes, "-"), pairs[:, 1].astype(np.str_))
