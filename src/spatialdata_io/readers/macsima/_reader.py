"""Assemble processed MACSima channel exports into lazy images and channel tables."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from spatialdata import SpatialData

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS, make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.macsima._channels import make_channel_options, plan_stack
from spatialdata_io.readers.macsima._images import make_image
from spatialdata_io.readers.macsima._layout import discover_layout, validate_layout_options
from spatialdata_io.readers.macsima._metadata import ChannelSource, read_channel
from spatialdata_io.readers.macsima._tables import make_channel_table

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence
    from pathlib import Path

    from anndata import AnnData
    from xarray import DataArray, DataTree

    from spatialdata_io.readers.macsima._layout import ParsingStyle, StackFiles


_STACK_CONTEXT = ReaderErrorContext("macsima", "channel stack")
_LAYOUT_CONTEXT = ReaderErrorContext("macsima", "layout")


def _read_sources(stack: StackFiles) -> list[ChannelSource]:
    """Inspect every candidate once; TIFFs without OME-XML are not channels."""
    sources: list[ChannelSource] = []
    skipped: list[str] = []
    for path in stack.files:
        source = read_channel(path)
        if source is None:
            skipped.append(path.name)
        else:
            sources.append(source)

    if skipped:
        warnings.warn(
            f"Skipping TIFF files without OME metadata in stack {stack.name!r}: {skipped}.", UserWarning, stacklevel=3
        )
    if not sources:
        raise ReaderFormatError(
            context=_STACK_CONTEXT,
            path=stack.directory,
            found=f"stack {stack.name!r} without OME-TIFF channels",
            expected="at least one MACSima OME-TIFF channel",
        )
    return sources


def _insert_unique[T](elements: MutableMapping[str, T], name: str, element: T, *, stack: StackFiles) -> None:
    """Insert one element and reject a name that another stack already generated.

    SpatialData element names must be unique ignoring case.
    """
    if any(existing.casefold() == name.casefold() for existing in elements):
        raise ReaderFormatError(
            context=_LAYOUT_CONTEXT,
            path=stack.directory,
            found=f"element name {name!r} generated twice, ignoring case",
            expected="folder names whose image and table names do not collide",
        )
    elements[name] = element


def macsima(  # noqa: PLR0913, RUF100 - Keyword-only options mirror the documented CLI.
    path: str | Path,
    *,
    parsing_style: ParsingStyle = "processed_single_folder",
    exclude_folder_names: Sequence[str] = (),
    exclude_cycles: Sequence[int] = (),
    include_cycle_in_channel_names: bool = False,
    nuclei_channel_name: str = "DAPI",
    split_threshold_nuclei_channel: int | None = 2,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
) -> SpatialData:
    """Read processed MACSima (MACSiQView) single-channel OME-TIFF exports.

    Every channel is one single-plane OME-TIFF; file names are not interpreted.
    Channel metadata comes from OME map annotations written by MACSiQView
    software major 0 or 1, from the OME plate well and, for major 0, from the
    screen reagent. A single-folder layout stacks every TIFF below ``path``::

        path/
        ├── <channel>.tif
        └── <subfolder>/<channel>.ome.tif

    A multiple-folder layout makes one stack per immediate subfolder::

        path/
        ├── <stack_a>/<channel>.tif
        └── <stack_b>/<channel>.tiff

    TIFF files are found recursively by a case-insensitive ``.tif``/``.tiff``
    suffix. Paths beginning with ``.`` and ``__MACOSX`` are ignored, and TIFFs
    without OME-XML are skipped with a warning.

    Parameters
    ----------
    path
        Processed MACSima folder.
    parsing_style
        ``"processed_single_folder"`` returns one stack;
        ``"processed_multiple_folders"`` returns one stack per subfolder.
        Subfolders without TIFFs are skipped with a warning.
    exclude_folder_names
        Exact subfolder names omitted in multiple-folder mode. Each name must
        exist; the option is invalid in single-folder mode.
    exclude_cycles
        Acquisition cycles to omit, exactly as recorded in the metadata.
    include_cycle_in_channel_names
        Name channels ``"R<cycle> <name>"`` instead of ``"<name>"``.
    nuclei_channel_name
        Exact marker name of the nuclear stain, for example ``"DAPI"``.
    split_threshold_nuclei_channel
        When a stack has more stain channels named ``nuclei_channel_name``
        than this number, all of them also form the separate nuclei stack and
        the main stack keeps only the first and the last (the registration
        reference and the end-of-run image). None never splits.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        images; empty for a single scale.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes of the
        images. By default one channel per chunk and square 1024-pixel tiles.

    Returns
    -------
    spatialdata.SpatialData
        For each stack named after its folder (made a valid element name):

        - ``<stack>_image``: a lazy ``(c, y, x)`` image in source dtype. Channels
          are ordered by cycle, then name, then file path. Names are the marker
          (bleach images add ``_background``), optionally prefixed with the
          cycle; repeated names are numbered in that order, for example
          ``DAPI (1)`` and ``DAPI (2)``.
        - ``<stack>_table``: ``AnnData`` with ``X=None``, no observations, and
          one ``var`` row per image channel (``var_names`` are the channel
          names). Columns: ``name`` (the marker, with ``_background`` for
          bleach images) and ``imagetype`` (``stain``, ``bleach`` or
          ``autofluorescence``) strings; ``cycle`` int64; ``well``,
          ``fluorophore`` and ``clone`` strings and ``ROI`` nullable Int64,
          NA when absent; ``exposure`` float64 milliseconds, NaN when absent.
          The table annotates no element.
        - ``<stack>_nuclei_image`` and ``<stack>_nuclei_table`` when the
          nuclei stack is split off. A channel has the same name in both
          images.

        One selected stack uses ``global`` in micrometers through the isotropic
        OME pixel size (µm or nm). Multiple independent stacks use their own
        ``<stack>`` frames; each nuclei stack shares its parent frame. Channels
        are placed on one canvas by their integral
        OME plane position in pixels (absent positions are the origin). The
        canvas starts at the smallest position and spans every channel;
        uncovered pixels are zero, with a warning. A position shared by all
        channels is not represented in the transformation. The reader name
        and, when all channels agree, the MACSiQView software version are
        recorded in ``attrs``.

    Raises
    ------
    FileNotFoundError, NotADirectoryError
        ``path`` is missing or not a directory.
    ValueError
        No channel TIFFs are found; element names collide; a TIFF or its
        OME-XML is malformed; OME content violates the v0/v1 contract
        (missing marker, cycle or acquisition type, conflicting annotation
        values, unsupported software version, units or plane positions, more
        than one plane); the channels of a stack differ in dtype, pixel size,
        software major, well or ROI; or option values are invalid, including
        ``exclude_cycles`` removing all channels of a stack and
        ``scale_factors`` exceeding the image.
    TypeError
        Public options have invalid types.

    Notes
    -----
    Each TIFF is inspected once. Construction executes no pixel tasks. Strips
    of whole rows are read in bands of at most 16 MiB, so computing a small
    region decodes only the intersecting bands. Source files must remain
    available. Vendor-written pyramid levels are not used.
    """
    raster = make_raster_options(scale_factors, chunks)
    channel_options = make_channel_options(
        exclude_cycles=exclude_cycles,
        include_cycle_in_channel_names=include_cycle_in_channel_names,
        nuclei_channel_name=nuclei_channel_name,
        split_threshold_nuclei_channel=split_threshold_nuclei_channel,
    )
    excluded_folders = validate_layout_options(parsing_style, exclude_folder_names)
    layout = discover_layout(path, parsing_style=parsing_style, exclude_folder_names=excluded_folders)

    images: dict[str, DataArray | DataTree] = {}
    tables: dict[str, AnnData] = {}
    versions: set[str] = set()
    for stack in layout.stacks:
        plan = plan_stack(stack, _read_sources(stack), channel_options)

        # The optional nuclei stack shares the main stack's canvas and frame.
        selections = {"": plan.channels}
        if plan.nuclei is not None:
            selections["_nuclei"] = plan.nuclei

        for suffix, channels in selections.items():
            image = make_image(
                channels,
                frame="global" if len(layout.stacks) == 1 else plan.name,
                pixel_size_um=plan.pixel_size_um,
                raster=raster,
            )
            _insert_unique(images, f"{plan.name}{suffix}_image", image, stack=stack)
            _insert_unique(tables, f"{plan.name}{suffix}_table", make_channel_table(channels), stack=stack)
            versions.update(channel.source.software_version for channel in channels)

    result = SpatialData(images=images, tables=tables)
    # Channels from different MACSiQView releases have no single format version.
    format_version = next(iter(versions)) if len(versions) == 1 else None
    set_reader_provenance(result.attrs, reader="macsima", format_version=format_version)
    return result
