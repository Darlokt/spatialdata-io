"""Assemble processed PhenoCycler and other QPI scans into a SpatialData dataset.

Each selected region contributes a lazy ``<region>_image`` in channel, row,
column order. Inspecting the primary scans together establishes their
coordinate systems. A requested ordinary 8-bit companion contributes a
second image on the primary scan's grid once the pair has been validated.

The region entries in ``attrs["akoya"]["regions"]`` preserve measured channel
metadata and summaries of any acquisition sidecars. Reader provenance
records the QPI description version when it is known and common to every
returned scan. Later pixel reads reopen the original TIFF paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from spatialdata import SpatialData

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.phenocycler._images import inspect_scan, read_scan, scan_attributes
from spatialdata_io.readers.phenocycler._layout import discover_layout, normalize_regions
from spatialdata_io.readers.phenocycler._sidecars import read_sidecars
from spatialdata_io.readers.phenocycler._transforms import scan_transformations, validate_companion

if TYPE_CHECKING:
    from pathlib import Path

_CONTEXT = ReaderErrorContext("phenocycler", "QPI scan")


def _normalize_channel_names(channel_names: Sequence[str] | None) -> tuple[str, ...] | None:
    if channel_names is None:
        return None
    if isinstance(channel_names, str | bytes) or not isinstance(channel_names, Sequence):
        message = "channel_names must be a sequence of strings."
        raise TypeError(message)
    if any(not isinstance(name, str) for name in channel_names):
        message = "channel_names must contain strings."
        raise TypeError(message)
    if (
        not channel_names
        or any(not name.strip() for name in channel_names)
        or len(set(channel_names)) != len(channel_names)
    ):
        message = "channel_names must contain unique, nonempty names."
        raise ValueError(message)
    return tuple(channel_names)


def phenocycler(  # noqa: PLR0913 - explicit reader options share the repository raster defaults.
    path: str | Path,
    *,
    regions: Sequence[str] | None = None,
    channel_names: Sequence[str] | None = None,
    read_all_bit_depths: bool = False,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read processed PhenoCycler-Fusion, CODEX Instrument Manager, or other Akoya QPI imagery.

    Supports Fusion 1.x and 2.x (including Opal), CODEX Instrument Manager,
    Vectra/PhenoImager RGB brightfield and fluorescence, and inForm floating
    point components in the PerkinElmer-QPI TIFF container. A source directory
    can contain several processed `<scan>.qptiff` files and optional
    `<scan>.er.qptiff`, `<scan>.xpd`, and `<scan>.channels.csv` artifacts.
    A HuBMAP deposit can store scans in `raw/images/` and channel annotations
    in `lab_processed/images/`. A generic `qptiff.channels.csv` applies to a
    single scan including its bit-depth variants. A directory with several
    scans uses scan-specific sidecar names. Directory discovery omits hidden
    files, AppleDouble artifacts, Zarr stores, and per-cycle acquisition files.

    Parameters
    ----------
    path
        Processed `.qptiff` or QPI `.tif`/`.tiff` file, or a directory of
        processed QPTIFFs. Sidecars for an explicit file are searched in its
        parent directory.
    regions
        Names of scan regions to return in order. Defaults to deterministic
        path order. Names come from a CODEX `regNNN` token or the scan stem,
        with a trailing `_Scan<N>` removed when possible without merging names.
    channel_names
        Optional unique, nonempty channel coordinates shared across all
        selected images. There must be one name per channel. Defaults to
        QPI `Biomarker`, then `Name`, then `channel_<zero-based position>`.
        Repeated source names receive suffixes ` (2)`, ` (3)`, and so on in
        page order, skipping spellings already used by a source channel.
        Declared RGB samples use `r`, `g`, and `b` by default.
    read_all_bit_depths
        Directory discovery prefers `.er.qptiff` over the ordinary file.
        An explicit file is read as selected. Enabling this option looks for
        its counterpart and also returns the ordinary variant as
        `<region>_8bit_image`. A pair needs unsigned 8-bit and higher-bit-depth
        samples, matching channel identities and shapes, and agreement on
        slide, calibration, and stage metadata. The counterpart uses the
        primary image's transformations.
    scale_factors
        Integer downsampling factors of at least two for generated pyramid
        levels. Empty for a single scale. The levels derive from the
        full-resolution image using SpatialData's image model.
    chunks
        Positive integer chunk size or three sizes in `c, y, x` order.
        Defaults to one channel and 1024 by 1024 pixels per chunk. The
        source graph uses TIFF-native geometry before rechunking.
    path_policy
        `"strict"` keeps discovered and relative sources inside the dataset
        directory or explicit file's parent directory. `"relaxed"` also
        permits external symlink targets. An explicitly supplied absolute file
        path may refer to a target elsewhere under either policy.
        Directory symlink ancestry cycles always raise an error.

    Returns
    -------
    SpatialData
        Images named `<region>_image` with `c, y, x` axes. Full-resolution
        channel planes follow physical TIFF page order and keep their stored
        values and dtypes, including unsigned integer and floating point
        signal. RGB samples become the leading channel axis. Thumbnail,
        overview, label, and reduced-resolution TIFF pages are excluded.

        A calibrated single scan uses `global` in micrometres. An uncalibrated
        single scan uses `image_pixels`. Multiple scans use `global` when
        they share a slide identifier and have calibrated grids and measured
        stage positions. Other groups have independent `<region>_um` or
        `<region>_pixels` frames. The scale comes from positive, consistent
        square-pixel TIFF centimetre resolutions. Nominal XML resolution is
        retained as metadata. The 96 dpi display fallback supplies no
        physical scale. When present, the TIFF X and Y stage positions locate
        the upper-left pixel edge in the corresponding physical frame.

        `attrs["akoya"]["regions"]` records the acquisition generation, QPI
        description version, selected channel measurements, calibration, and
        optional sidecar summaries. Pixel sizes are X then Y in micrometres
        per pixel. Stage positions are X then Y in micrometres. The page
        exposure is in microseconds and supported protocol exposures are in
        milliseconds. The TIFF sample minimum and maximum describe stored
        pixel values. Signal unit codes and normalization factors describe
        the acquisition and do not rescale the data.

        An XPD protocol's cycle and distinct marker counts and a channel CSV's
        optional `Marker` list are summarized there. HuBMAP annotation files
        supply one row per channel with a unique `channel id` and Yes/No
        flags for nuclei segmentation, cell segmentation, and antibody
        status. They retain source order under `channels_csv["annotations"]`.
        Sidecar annotations leave the channel identities from QPI unchanged.

        Image pixels remain lazy, and computing a slice reads the TIFF tiles
        or scanline bands needed by the requested output chunks. The files
        close on return and must remain available for later reads. The
        function leaves caller-owned options unchanged.

    Warns
    -----
    UserWarning
        If file names need sanitizing or sidecar marker names differ from
        the QPI channel names.

    Raises
    ------
    TypeError, ValueError
        If a reader option is invalid or requested regions are unavailable.
    OSError
        If an artifact is missing, has the wrong kind, or cannot be opened.
    ReaderFormatError
        If a layout, QPI scan, calibration, bit-depth pair or sidecar is
        ambiguous or malformed. A supplied XPD or channel sidecar is
        limited to 4 MiB.
    """
    selected_regions = normalize_regions(regions)
    names = _normalize_channel_names(channel_names)
    if not isinstance(read_all_bit_depths, bool):
        message = "read_all_bit_depths must be a boolean."
        raise TypeError(message)
    options = make_raster_options(scale_factors, chunks)
    layout = discover_layout(
        path, regions=selected_regions, read_all_bit_depths=read_all_bit_depths, path_policy=path_policy
    )
    scans = {files.scan.region: inspect_scan(files.scan.image, context=_CONTEXT) for files in layout.regions}
    transformations = scan_transformations(scans)
    region_attrs = {}
    for files in layout.regions:
        region = files.scan.region
        scan = scans[region]
        region_attrs[region] = scan_attributes(scan, names) | read_sidecars(
            files, (channel.source_name for channel in scan.channels), detected_version=scan.format_version
        )
        if read_all_bit_depths and files.scan.lower_bit_depth is not None:
            companion = inspect_scan(files.scan.lower_bit_depth, context=_CONTEXT)
            validate_companion(scan, companion, context=_CONTEXT)
            companion_region = f"{region}_8bit"
            scans[companion_region] = companion
            transformations[companion_region] = transformations[region]
            region_attrs[companion_region] = scan_attributes(companion, names) | {"primary_region": region}

    images = {
        f"{region}_image": read_scan(
            scan, options=options, channel_names=names, transformations=transformations[region], context=_CONTEXT
        )
        for region, scan in scans.items()
    }
    result = SpatialData(images=images, attrs={"akoya": {"regions": region_attrs}})
    versions = {scan.format_version for scan in scans.values()}
    set_reader_provenance(
        result.attrs, reader="phenocycler", format_version=next(iter(versions)) if len(versions) == 1 else None
    )
    return result
