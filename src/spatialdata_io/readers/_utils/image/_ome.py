"""Interpret OME pixel sizes and channel identities declared in an inspected TIFF.

OME-XML describes each image's ``Pixels`` with optional physical sizes and
their length units, and one ``Channel`` record per channel with a required ID,
an optional name and an optional number of samples per pixel::

    <Pixels PhysicalSizeX="0.17" PhysicalSizeXUnit="µm" SizeC="2" ...>
      <Channel ID="Channel:0:0" Name="DAPI" SamplesPerPixel="1"/>
      ...

An interleaved RGB image is one ``Channel`` with ``SamplesPerPixel="3"``
(``SizeC`` counts samples), which ``tifffile`` exposes as a samples axis
``S`` of length three rather than as a channel axis ``C``.

These helpers normalize those declarations and compare two of them; readers
decide what they calibrate. ``raster_channel_names`` decides when the OME
identifiers label a raster's channels: a colour model declared by the TIFF
itself (``r, g, b``) takes precedence over them. Schema:
https://www.openmicroscopy.org/Schemas/OME/2016-06
"""

from __future__ import annotations

from math import isclose, isfinite
from typing import TYPE_CHECKING, Literal

from spatialdata_io.readers._utils.errors import ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.image.tiff import TiffMetadata

# (multiplier, divisor) to micrometres. Integer factors make each conversion
# one correctly rounded operation: 9 nm becomes exactly the float nearest
# 0.009 µm, which multiplying by the inexact 0.001 does not guarantee.
_TO_MICROMETERS: dict[str, tuple[int, int]] = {"µm": (1, 1), "nm": (1, 1000), "mm": (1000, 1)}

# Declarations of one pixel size reach a reader through different arithmetic
# (an exact extent/width ratio, a producer's float division, decimal OME text,
# a unit conversion), which can differ in the last bits. This relative
# tolerance absorbs that and is far below any physically meaningful
# disagreement; it deliberately does not absorb a float32 round trip (about
# 6e-8 relative).
PIXEL_SIZE_RTOL = 1e-12


def ome_pixel_sizes_um(
    metadata: TiffMetadata,
    *,
    series: int,
    axes: tuple[Literal["x", "y", "z"], ...],
    context: ReaderErrorContext,
) -> tuple[float | None, ...]:
    """Return each requested declared OME pixel length in micrometres.

    Parameters
    ----------
    metadata
        Borrowed, read-only metadata of one inspected TIFF.
    series
        Selected TIFF series; inspection guarantees one OME image per series.
    axes
        Spatial axes whose declarations are requested, in output order.
    context
        Owning reader and artifact for malformed declaration errors.

    Returns
    -------
    tuple of float or None
        One entry per requested axis: a finite positive length in micrometres,
        or ``None`` when that size is not declared. Every entry is ``None``
        without OME metadata.

    Raises
    ------
    ReaderFormatError
        A present size or its unit violates the supported length contract,
        even when other requested sizes are absent.

    Notes
    -----
    Sizes in µm, nm and mm are supported; each conversion is one correctly
    rounded operation, so 9 nm gives exactly ``0.009``. The reader decides whether a partial
    declaration calibrates anything and owns axis association, dimensions,
    agreement between artifacts, registration and transformation assembly.
    No source metadata is changed and no pixel data is read.
    """
    if metadata.ome is None:
        return (None,) * len(axes)
    pixels = metadata.ome.images[series].pixels
    fields = {
        "x": (pixels.physical_size_x, pixels.physical_size_x_unit),
        "y": (pixels.physical_size_y, pixels.physical_size_y_unit),
        "z": (pixels.physical_size_z, pixels.physical_size_z_unit),
    }
    result: list[float | None] = []
    for axis in axes:
        value, unit = fields[axis]
        if value is None:
            result.append(None)
            continue
        factors = _TO_MICROMETERS.get(unit.value)
        # A NaN, infinite or nonpositive size, or one overflowing or
        # underflowing on conversion, fails the single check on the result.
        micrometres = None if factors is None else value * factors[0] / factors[1]
        if micrometres is None or not isfinite(micrometres) or micrometres <= 0:
            raise ReaderFormatError(
                context=context,
                path=metadata.path,
                found=f"PhysicalSize{axis.upper()}={value!r} {unit.value}",
                expected="finite positive physical sizes in µm, nm or mm, representable in µm",
            )
        result.append(micrometres)
    return tuple(result)


def require_matching_pixel_sizes(  # noqa: PLR0913, RUF100 - Two declarations plus explicit error context.
    declared: Sequence[float | None],
    reference: Sequence[float | None],
    /,
    *,
    axes: str,
    context: ReaderErrorContext,
    path: Path,
    expected: str,
) -> None:
    """Require two pixel-size declarations of one grid to agree wherever both are present.

    Parameters
    ----------
    declared
        Sizes in micrometres declared by the artifact being checked, one per
        axis of ``axes``; ``None`` where a size is absent.
    reference
        Sizes in micrometres of the grid it must share, in the same order.
    axes
        One letter per entry, such as ``"xy"``, used in the message.
    context
        Owning reader and artifact of ``declared``.
    path
        File that declared ``declared``.
    expected
        Reader-owned description of the required agreement.

    Raises
    ------
    ReaderFormatError
        If two present sizes of one axis differ by more than
        ``PIXEL_SIZE_RTOL`` relative.

    Notes
    -----
    An absent size never conflicts: the caller inherits it from the other
    declaration. Which declaration calibrates the grid, and whether a partial
    one calibrates anything, remain reader decisions.
    """
    for axis, size, reference_size in zip(axes, declared, reference, strict=True):
        if size is None or reference_size is None or isclose(size, reference_size, rel_tol=PIXEL_SIZE_RTOL):
            continue
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"{axis.upper()} pixel size {size} versus {reference_size} µm",
            expected=expected,
        )


def ome_channel_names(
    metadata: TiffMetadata,
    *,
    series: int,
    count: int,
    samples: bool,
    context: ReaderErrorContext,
) -> list[str] | None:
    """Return one unique OME identifier per raster channel.

    Parameters
    ----------
    metadata
        Borrowed, read-only metadata of one inspected TIFF.
    series
        Selected TIFF series; inspection guarantees one OME image per series.
    count
        Number of channels of the reader's normalized raster.
    samples
        Whether the raster's channel axis is the TIFF samples axis ``S``.
        OME declares an interleaved RGB image as one ``Channel`` with
        ``SamplesPerPixel="3"``, so each OME channel then accounts for
        ``SamplesPerPixel`` raster channels. Otherwise the channel axis is the
        TIFF channel axis ``C``, or a length-one axis added by the reader, with
        one OME channel per raster channel.
    context
        Owning reader and artifact for malformed declaration errors.

    Returns
    -------
    list of str or None
        ``None`` without OME metadata. Otherwise each OME channel is
        identified by its name when all names are nonempty and unique, else by
        its OME ID. Without ``samples`` that identifier names its raster
        channel. With ``samples`` the channel expands, in OME order, to
        ``"<identifier>:sample:<index>"`` for each zero-based sample index, so
        an unnamed RGB channel gives ``["Channel:0:0:sample:0",
        "Channel:0:0:sample:1", "Channel:0:0:sample:2"]``.

    Raises
    ------
    ReaderFormatError
        If the OME channels (or, with ``samples``, their samples) do not
        number ``count``, or neither their names nor their IDs are unique.

    Notes
    -----
    Sample identifiers keep the source channel record and the sample order
    without assuming a colour interpretation; pass the result to
    ``raster_channel_names``, which labels samples ``r, g, b(, a)`` instead
    when the TIFF declares an RGB colour model. They are unique whenever the
    channel identifiers are, because the identifier is everything before the
    last ``":sample:"``. An absent ``SamplesPerPixel`` counts as one sample.
    """
    if metadata.ome is None:
        return None
    channels = metadata.ome.images[series].pixels.channels
    if samples:
        sizes = [channel.samples_per_pixel or 1 for channel in channels]
        found = f"{len(channels)} OME channels with {sum(sizes)} samples for {count} raster channels"
        expected = "one OME channel sample per raster channel of the TIFF samples axis"
    else:
        sizes = [1] * len(channels)
        found = f"{len(channels)} OME channels for {count} raster channels"
        expected = "one OME channel record per raster channel"
    if sum(sizes) != count:
        raise ReaderFormatError(context=context, path=metadata.path, found=found, expected=expected)

    names = [channel.name for channel in channels]
    if all(names) and len(set(names)) == len(names):
        identifiers = [str(name) for name in names]
    else:
        identifiers = [channel.id for channel in channels]
        if len(set(identifiers)) != len(identifiers):
            raise ReaderFormatError(
                context=context,
                path=metadata.path,
                found=f"duplicate OME channel names and IDs {identifiers!r}",
                expected="unique nonempty OME channel names or unique OME channel IDs",
            )
    if not samples:
        return identifiers
    return [
        f"{identifier}:sample:{index}"
        for identifier, size in zip(identifiers, sizes, strict=True)
        for index in range(size)
    ]
