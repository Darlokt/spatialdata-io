"""Label the channels of a raster from the colour model its source declares.

SpatialData names the channels of a colour image ``r, g, b`` or ``r, g, b, a``
so that viewers composite them. A raster earns these names only from its own
file's declaration, never from its channel count, dtype or SpatialData's
``rgb=None`` guessing. The rule, applied in order, is:

1. The raster's channel axis is the source's interleaved samples axis ``s`` and
   the source declares a colour model for those samples (``LazyRaster.color``
   or ``TiffLevelMetadata.color``): ``r, g, b``, or ``r, g, b, a`` when the
   fourth sample is declared alpha.

   - TIFF: ``PhotometricInterpretation`` RGB, with ``ExtraSamples`` declaring
     associated or unassociated alpha for a fourth sample, or JPEG-compressed
     YCbCr, which the codec decodes to RGB.
   - PNG: colour types 2 (RGB), 3 (palette of RGB entries) and 6 (RGBA); a
     ``tRNS`` chunk adds alpha. Grayscale and grayscale + alpha declare none.
   - JPEG: three components, which the decoder returns as RGB.

2. Otherwise, the OME channel identifiers when the raster's channel axis
   corresponds to the OME channels, from ``ome_channel_names``: names or IDs,
   or ``<name or ID>:sample:<k>`` for non-RGB multi-sample OME data.

3. Otherwise ``None``, and SpatialData assigns positional ``0..n-1``.

Readers decide which source axis becomes the channel axis, from the file's
declared axes or a caller's ``data_axes``. ``channel_source`` then derives,
from that correspondence alone, whether the channel axis is the samples axis
and whether the OME channel records describe it, so a reinterpreted axis never
receives OME names that describe another axis. This module owns that
correspondence, the precedence and the colour names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spatialdata_io.readers._utils.image._types import SampleColor


def channel_source(declared: Sequence[str], source: Sequence[str]) -> tuple[bool, bool | None]:
    """Return how the raster's channel axis corresponds to the file's samples and OME channels.

    Parameters
    ----------
    declared
        Lowercase axes of the selected level as inspected. Inspection drops
        every singleton axis except ``y`` and ``x``, so a declared ``c`` or
        ``s`` is never singleton.
    source
        The reader's lowercase reading of the same axes, in the same order:
        ``declared`` itself or a caller's ``data_axes``. ``c`` and ``s`` both
        mark the channel axis; after axis normalization accepted ``source``
        there is at most one, and without one the reader added a singleton
        channel.

    Returns
    -------
    tuple of bool and bool or None
        ``(samples, ome_samples)``. ``samples`` is whether the channel axis is
        the declared samples axis, the argument of
        :func:`raster_channel_names`. ``ome_samples`` is the ``samples``
        argument of ``ome_channel_names``, or ``None`` when the OME channel
        records do not describe the channel axis and must not name it:

        ========================================  =======  ===========
        channel axis of ``source``                samples  ome_samples
        ========================================  =======  ===========
        declared ``c``                            False    False
        declared ``s``, no declared ``c``         True     True
        declared ``s`` beside a declared ``c``    True     None
        any other declared axis                   False    None
        added; no declared ``c`` or ``s``         False    False
        added; a declared ``c`` or ``s``          False    None
        ========================================  =======  ===========

    Raises
    ------
    ValueError
        If ``declared`` and ``source`` differ in length.

    Notes
    -----
    OME declares an interleaved RGB image as one channel record with several
    samples. Beside a separate ``c`` axis the records describe that axis, not
    the samples, and an added singleton channel is one OME channel only when
    the file has no channel or samples axis at all.
    """
    channel = next((axis for axis, role in zip(declared, source, strict=True) if role in {"c", "s"}), None)
    if channel is None:
        return False, None if {"c", "s"}.intersection(declared) else False
    if channel == "c":
        return False, False
    if channel == "s":
        return True, None if "c" in declared else True
    return False, None


def raster_channel_names(
    color: SampleColor | None,
    *,
    samples: bool,
    ome: list[str] | None,
) -> list[str] | None:
    """Return the channel coordinates of a raster under the shared labelling rule.

    Parameters
    ----------
    color
        The colour model the source declares for its samples axis, from
        ``LazyRaster.color`` or ``TiffLevelMetadata.color`` of the selected
        level.
    samples
        Whether the raster's channel axis is the source's samples axis ``s``
        (TIFF ``S``, PNG or JPEG samples).
    ome
        OME channel identifiers from ``ome_channel_names`` when the
        channel axis corresponds to the OME channels, else ``None``.

    Returns
    -------
    list of str or None
        ``["r", "g", "b"]`` or ``["r", "g", "b", "a"]`` when ``samples`` and a
        declared ``color``; otherwise ``ome``, where ``None`` means positional
        channel coordinates.

    Notes
    -----
    A declared colour model always has one sample per letter: inspection sets
    it only when the samples axis has exactly three or four entries. OME
    identifiers that a declared colour model supersedes were still validated
    by ``ome_channel_names``, so corrupt OME channel metadata fails in
    either case.
    """
    if samples and color is not None:
        # Each letter of the declared model names one sample, in order.
        return list(color)
    return ome
