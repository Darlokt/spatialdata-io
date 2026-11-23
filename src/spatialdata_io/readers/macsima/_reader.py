"""Assemble processed MACSima channel exports into lazy images and channel tables.

The public function only orchestrates: validate the options in signature
order, discover the stacks, inspect every candidate TIFF once, plan each
stack's channels, then build one image and one metadata table per stack (and
per split nuclei stack) and record provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from spatialdata import SpatialData

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, warn_user
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.names import colliding_element_names
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.macsima._channels import make_channel_options, plan_stack
from spatialdata_io.readers.macsima._images import make_image
from spatialdata_io.readers.macsima._layout import discover_layout, validate_layout_options
from spatialdata_io.readers.macsima._metadata import ChannelSource, read_channel
from spatialdata_io.readers.macsima._tables import make_channel_table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from xarray import DataArray, DataTree

    from spatialdata_io.readers.macsima._channels import PlannedChannel
    from spatialdata_io.readers.macsima._layout import ParsingStyle, StackFiles


_STACK_CONTEXT = ReaderErrorContext("macsima", "channel stack")
_LAYOUT_CONTEXT = ReaderErrorContext("macsima", "dataset layout")


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
        warn_user(f"macsima: skipping TIFF files without OME metadata in stack {stack.name!r}: {skipped}.")
    if not sources:
        raise ReaderFormatError(
            context=_STACK_CONTEXT,
            path=stack.directory,
            found=f"stack {stack.name!r} without OME-TIFF channels",
            expected="at least one MACSima OME-TIFF channel",
        )
    return sources


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
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read processed MACSima (MACS iQ View) single-channel OME-TIFF exports.

    Supported layouts; file names are never interpreted::

        run/                              experiment/
        ├── <channel>.tif                 ├── <stack_a>/<channel>.tif
        └── <subfolder>/<channel>.tif     └── <stack_b>/<channel>.tiff

    - ``"processed_single_folder"`` stacks every TIFF below ``path``;
      ``"processed_multiple_folders"`` makes one stack per immediate
      subfolder and ignores files in ``path`` itself.
    - TIFFs have a case-insensitive ``.tif`` or ``.tiff`` suffix. Paths
      starting with ``.`` and ``__MACOSX`` are ignored, TIFFs without
      OME-XML are skipped with a warning, and a dangling symlink fails.
    - Every channel file holds one plane. The marker, cycle, acquisition
      type, ROI, fluorophore, clone and exposure come from the OME map
      annotations of MACS iQ View software major 0 or 1, the well from the
      OME plate and, for major 0, marker and clone from the screen reagent.
    - Each stack is named after its folder. A folder name that is not a
      valid SpatialData element name is repaired with a warning (``my run``
      becomes ``my_run``); folders whose names then agree ignoring case fail.

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
        Non-negative acquisition cycles to omit, as recorded in the metadata.
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
    path_policy
        How symlinks inside the dataset directory are treated. ``"strict"``
        requires every discovered artifact and relative file option to resolve
        inside the directory. ``"relaxed"`` also accepts symlinks to targets
        stored elsewhere, as in git-annex or DVC datasets. Relative paths may
        not leave the directory with ``..`` in either mode, and absolute file
        options may point anywhere. File options keep the name the caller
        gave, and relaxed reads keep each artifact's own name rather than that
        of its link target.

    Returns
    -------
    spatialdata.SpatialData
        Per stack, with ``<stack>`` its (repaired) folder name:

        - ``images["<stack>_image"]``: lazy ``(c, y, x)`` image in the source
          dtype. Channels are ordered by cycle, then name, then discovered file path, and
          named after the marker (bleach images add ``_background``),
          optionally prefixed with the cycle. Repeated names are numbered in
          that order, for example ``DAPI (1)`` and ``DAPI (2)``.
        - ``tables["<stack>_table"]``: ``AnnData`` with ``X=None``, no
          observations and one ``var`` row per image channel; ``var_names``
          are the channel names and the table annotates no element. Columns:
          ``name`` (the marker, with ``_background`` for bleach images) and
          ``imagetype`` (``stain``, ``bleach`` or ``autofluorescence``)
          strings; ``cycle`` uint64; ``well``, ``fluorophore`` and ``clone``
          strings and ``ROI`` nullable UInt64, NA when absent; ``exposure``
          float64 milliseconds, NaN when absent. Cycles and ROI IDs range
          from zero through ``2**64 - 1`` and retain exact values in H5AD
          and Zarr storage.
        - ``images["<stack>_nuclei_image"]`` and
          ``tables["<stack>_nuclei_table"]`` when the nuclei stack is split
          off. A channel has the same name in both images.

        ``attrs`` record the reader, the package version and, when all
        returned channels agree, the MACS iQ View software version as the
        format version.

    Raises
    ------
    TypeError
        If an option has the wrong type.
    ValueError
        If an option has an invalid value, including ``exclude_folder_names``
        that match no folder, ``exclude_cycles`` removing every channel of a
        stack and ``scale_factors`` exceeding the image.
    FileNotFoundError, NotADirectoryError
        If ``path`` is missing or not a directory, or a TIFF is a dangling
        symlink.
    ValueError
        A contextual ``ReaderFormatError`` naming the reader, artifact, path
        and, once known, the detected software version if no channel TIFF is
        found; element names collide; OME content violates the major 0 or 1
        contract (missing marker, cycle or acquisition type, conflicting
        annotation values, unsupported software version, units, plane
        positions or values, more than one plane); or the channels of a stack
        differ in dtype, pixel size, software major, well or ROI. A malformed
        TIFF or OME-XML raises ``RasterFormatError``.

    Warns
    -----
    UserWarning
        When a folder name is repaired into an element name, TIFFs without
        OME-XML or subfolders without TIFFs are skipped, or channels are
        zero-padded onto a common canvas.

    Notes
    -----
    Coordinates: every stack is calibrated in micrometres by its square OME
    pixel size, declared in µm or nm. Sizes must be positive and agree
    across axes and channels within a relative tolerance of ``1e-12``.
    The first selected channel's X size supplies the scale. One stack uses
    ``global``. Several stacks are
    not registered to each other, so each keeps its own micrometre frame
    named ``<stack>``; a nuclei stack shares its parent's frame. Channels are
    placed on one pixel canvas by their integral OME plane positions in
    pixels (absent positions are the origin). The canvas starts at the
    smallest position and spans every channel; uncovered pixels are zero.
    A position shared by all channels is not represented in the
    transformation.

    Laziness: each TIFF is inspected once and construction executes no pixel
    tasks. Source chunks follow native tiles or whole-row bands of about
    16 MiB of decoded pixels; a larger native strip stays one chunk, so a
    plane stored as one strip is decoded whole. Rechunking to ``chunks`` can
    make a small crop read neighbouring tiles or bands. Keep the files
    available until the pixels are computed or written. Vendor-written
    pyramid levels are not used.
    """
    excluded_folders = validate_layout_options(parsing_style, exclude_folder_names)
    channel_options = make_channel_options(
        exclude_cycles=exclude_cycles,
        include_cycle_in_channel_names=include_cycle_in_channel_names,
        nuclei_channel_name=nuclei_channel_name,
        split_threshold_nuclei_channel=split_threshold_nuclei_channel,
    )
    raster = make_raster_options(scale_factors, chunks)
    # Discovery validates path_policy before touching the filesystem.
    layout = discover_layout(
        path, parsing_style=parsing_style, exclude_folder_names=excluded_folders, path_policy=path_policy
    )

    selections: list[tuple[str, str, tuple[PlannedChannel, ...], float]] = []
    for stack in layout.stacks:
        plan = plan_stack(stack, _read_sources(stack), channel_options)
        # Stacks are calibrated in micrometres; unregistered stacks keep separate frames.
        frame = "global" if len(layout.stacks) == 1 else plan.name
        selections.append((plan.name, frame, plan.channels, plan.pixel_size_um))
        if plan.nuclei is not None:
            selections.append((f"{plan.name}_nuclei", frame, plan.nuclei, plan.pixel_size_um))

    # A split stack's name can coincide with an ordinary folder name.
    duplicates = colliding_element_names(f"{name}_image" for name, _, _, _ in selections)
    if duplicates:
        raise ReaderFormatError(
            context=_LAYOUT_CONTEXT,
            path=Path(path).expanduser(),
            found=f"generated element names {duplicates!r} that collide ignoring case",
            expected="folder names whose image and table names do not collide",
        )

    images: dict[str, DataArray | DataTree] = {}
    tables: dict[str, AnnData] = {}
    versions: set[str] = set()
    for name, frame, channels, pixel_size_um in selections:
        images[f"{name}_image"] = make_image(channels, frame=frame, pixel_size_um=pixel_size_um, raster=raster)
        tables[f"{name}_table"] = make_channel_table(channels)
        versions.update(channel.source.software_version for channel in channels)

    result = SpatialData(images=images, tables=tables)
    # Channels from different MACS iQ View releases have no single format version.
    format_version = next(iter(versions)) if len(versions) == 1 else None
    set_reader_provenance(result.attrs, reader="macsima", format_version=format_version)
    return result
