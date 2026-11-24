"""Assemble CODEX regional images, cell positions and expression tables.

A flat export can supply an image, a cell table or both. Named regions may
supply stitched Processor marker planes, a Nolan Uploader montage or a CIM
QPI scan. Each image keeps its source channel order and calibration according
to its layout. An explicit image-to-table transform registers a raster to
the table's native region pixels.

Each selected table carries sparse cycle/channel expression. Its cell IDs
annotate radius-one display circles at the measured centroids, or a supplied
instance mask registered to the regional image. Mask linkage preserves cell
IDs and their native table positions. All raster pixels remain lazy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from spatialdata import SpatialData
from spatialdata.transformations import Identity, Scale
from xarray import DataTree

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe
from spatialdata_io.readers.codex._calibration import image_transformation, resolve_calibration, validate_calibration
from spatialdata_io.readers.codex._geometry import make_shapes
from spatialdata_io.readers.codex._images import read_image, validate_image_options
from spatialdata_io.readers.codex._layout import MarkerSet, discover_layout, validate_region_options
from spatialdata_io.readers.codex._tables import read_cell_table, set_positions
from spatialdata_io.readers.phenocycler._images import inspect_scan, read_scan, scan_attributes
from spatialdata_io.readers.phenocycler._labels import read_mask
from spatialdata_io.readers.phenocycler._layout import ScanFiles
from spatialdata_io.readers.phenocycler._transforms import scan_transformations, validate_companion

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import pandas as pd
    from anndata import AnnData
    from geopandas import GeoDataFrame
    from spatialdata.transformations import Affine
    from xarray import DataArray

    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers.codex._calibration import Calibration
    from spatialdata_io.readers.codex._layout import CellTableSource, CodexLayout


def codex(  # noqa: PLR0913, RUF100 - explicit format choices keep the public boundary typed.
    path: str | Path,
    *,
    cell_table_file: str | Path | None = None,
    image_file: str | Path | None = None,
    regions: Sequence[str] | None = None,
    read_tables: bool = True,
    compensation: Literal["compensated", "uncompensated"] | None = None,
    pixel_size_um: float | None = None,
    experiment_file: str | Path | None = None,
    image_to_table: Affine | None = None,
    data_axes: str | None = None,
    tiff_series: int | None = None,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    labels: str | Path | Mapping[str, str | Path] | None = None,
    labels_to_image: Affine | Mapping[str, Affine] | None = None,
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read a flat CODEX region or one or more named CODEX regions.

    Supported layout (one region, unversioned processor output)::

        region/
        ├── reg001_compensated.csv  # or equivalent .fcs (FCS 3.1)
        ├── stitched.tif            # optional TIFF, OME-TIFF, BigTIFF or BTF
        └── experiment.json         # optional Akoya experiment, only read when selected

    A Processor experiment has a ``stitched/regNNN/`` directory containing
    ``regNNN_cycCCC_chHHH_<marker>.tif`` or ``..._chaHHH_<marker>.tif`` single
    planes and may contain regional tables in
    ``segm/segm-*/fcs/{compensated,uncompensated}/``. These directories may
    be inside one ``processed_*`` directory. Uploader stores ImageJ
    ``regNNN_montage.tif`` files under ``bestFocus/`` or
    ``bestFocus/stitched/``; named montages under ``stitched/`` also work.
    CODEX Instrument Manager QPTIFF scans can be supplied directly or in a
    directory. Segmentation mask files ending in ``_mask`` or ``_labels``
    are excluded from image discovery.

    The cell table needs the named fields ``cell_id``, ``region``, ``x``, ``y``
    and at least one ``cyc<digits>_ch<digits>`` expression field, in any order;
    all other fields are kept as observation metadata. Akoya CSV headers of the
    form ``short_name:long_label`` are supported when every field is qualified.
    Convert original Nolan cell tables to this column schema before reading.

    Parameters
    ----------
    path
        A directory or one TIFF, CIM QPTIFF, CSV, or FCS file. In direct-file
        mode sibling images and tables are not discovered, and the file's
        parent directory is used for relative overrides.
    cell_table_file
        Explicit CSV or FCS table to use for one selected region. Relative
        paths resolve against the dataset directory. Absolute paths may point
        elsewhere. Requires ``read_tables=True``.
    image_file
        Stitched image to read when the directory holds several. Resolved like
        ``cell_table_file``. Requires exactly one selected region.
    regions
        Distinct named regions to read, in output order. The default reads
        all named regions in lexical order. A flat region has no selector.
    read_tables
        Read available cell tables and build centroid circles for regions
        without supplied masks. Set to false to read only the requested rasters.
    compensation
        Select ``"compensated"`` or ``"uncompensated"`` tables by filename
        suffix for every selected region that has table candidates. A symlink
        supplies its own filename for this selection and the table metadata. With
        ``None``, both compensation variants are eligible. An eligible CSV/FCS
        pair with the same stem in the same directory is read from CSV to
        preserve precision. Other multiple matches are ambiguous. Requires
        ``read_tables=True``, at least one table and automatic table discovery.
    pixel_size_um
        Positive finite micrometres per native table pixel, representable as
        float64. Requires a table. The caller asserts that it belongs to
        each selected table.
    experiment_file
        Akoya experiment JSON (version ``1.5.0.38`` or ``1.7.0.6``) whose
        ``xyResolution`` in nm/pixel calibrates the table. It is never
        discovered automatically; selecting it asserts that it belongs to the
        table. Requires a table. Resolved like ``cell_table_file``. Together with
        ``pixel_size_um``, both must agree exactly (``xyResolution / 1000``).
    image_to_table
        Invertible finite XY ``Affine`` from image pixel coordinates (pixel
        centres at ``column + 0.5, row + 0.5``) to native table ``x, y``.
        With calibration, the composed float64 transform into ``global`` must
        remain finite and invertible.
        Without it the image is not registered to the table. Requires one
        region containing an image and a table.
    data_axes
        Source TIFF axes as one string of distinct letters including ``X`` and
        ``Y``, for example ``"CYX"`` for a stack without axis metadata; ``S``
        (samples) counts as ``C``. Defaults to the axes the file declares. An
        axis read as channels that the file declares otherwise gets positional
        channel names. Requires a TIFF image. QPTIFF scan layout is selected
        by its metadata.
    tiff_series
        Zero-based TIFF series. Required when the file has several series.
        Requires a TIFF image. QPTIFF scan layout is selected by its metadata.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        image; empty for a single scale. Nondefault values require an image.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes; by default
        one channel per chunk and square 1024-pixel tiles. Nondefault values
        require an image.
    labels
        Integer instance-label TIFF for one selected image, or a mapping from
        selected region names to files. Relative paths resolve against the
        input directory (the parent for a direct file). Supplying a mask
        asserts it segments that image. The key for a flat image is ``"images"``.
        On the same grid their YX extents must agree. Zero is background. Only singleton nonspatial axes are removed.
    labels_to_image
        Optional invertible XY affine mapping mask pixel coordinates into
        full-resolution image pixels. Supply the measured registration for
        cropped, rotated or resized masks. A mapping can specify different
        affines by mask region. An omitted affine declares the same grid.
    path_policy
        How symlinks inside the dataset directory are treated. ``"strict"``
        requires every discovered artifact and relative file option to resolve
        inside the directory. ``"relaxed"`` also accepts symlinks to targets
        stored elsewhere, as in git-annex or DVC datasets. Relative paths may
        not leave the directory with ``..`` in either mode, and absolute file
        options may point anywhere. The selected file name determines the
        table and image formats under either policy.

    Returns
    -------
    SpatialData
        One linked table per available cell population, and one
        lazy image per available image source. Flat layouts use ``"table"``
        and ``"images"``. Named layouts use ``"<region>_table"`` and
        ``"<region>_image"``.

        ``tables["table"]``
            One row per cell in source order; observation names are the
            zero-based row positions.

            - ``X``: CSR sparse array of the cycle intensities, in source
              feature order. CSV uses float64; FCS keeps its native dtype
              and rejects lossy integer promotion. Negative
              compensated intensities are kept. CSV is parsed in bounded
              batches, so the dense expression of the whole table never
              exists at once.
            - ``uns["spatialdata_io_compensation"]``: signal product when
              the table filename identifies compensated or uncompensated data.
            - ``var``: indexed by the cycle field names; ``var["marker"]``
              holds FCS ``$PnS`` labels or Akoya long labels when present.
            - ``obs``: every non-cycle field. ``cell_id`` is uint64
              (including zero) for CSV and FCS alike.
              ``region`` is categorical; CSV ``x`` and ``y`` are float64
              native pixels. Other CSV fields keep their exact text
              (including ``NA``, ``NULL``, leading zeros and empty strings);
              FCS fields keep their numeric dtypes. Field names that a
              SpatialData table does not accept (``tile num``) are sanitized
              with a warning (``tile_num``; a name that then collides,
              ignoring letter case, gets a ``_1``, ``_2``, ... suffix).
            - ``obsm["spatial"]`` (only with a calibration): float64 ``x, y``
              in micrometres, described in ``uns["spatialdata_io_positions"]``.

        ``labels["<region>_labels"]`` (flat layout: ``labels["labels"]``)
            Lazy YX integer instances on the registered image frame. The supplied
            table must identify cells by their positive mask values. The reader
            checks the dtype range and preserves the normalized region name in
            ``table.uns["spatialdata_io_source_region"]``. Membership is a caller
            assertion: collecting IDs would require reading the mask pixels.

        ``shapes[region]`` (without a supplied mask)
            Radius-one circles at the native cell positions, indexed by
            uint64 ``cell_id`` and linked to the table. The element is named after
            the region: literal CSV strings, or positive FCS integers as
            decimal strings. Names must be valid element names other than
            ``table`` and ``images``.
            The circles represent cell positions for display.
        ``images["images"]`` (only with an image)
            Lazy CYX image of the selected series at full resolution, with the
            source dtype and channel order. Channels are named by OME channel
            names when all are nonempty and unique, otherwise by OME channel
            IDs, otherwise by position; they are not matched to table markers.
            Interleaved samples become channels: ``r, g, b`` (``r, g, b, a``
            with a declared alpha sample) when the TIFF declares them RGB
            (``PhotometricInterpretation`` RGB or JPEG-compressed YCbCr);
            otherwise an OME channel with several samples names them
            ``<name or ID>:sample:0``, ``:sample:1``, and so on. A pyramid is
            built only when ``scale_factors`` asks for one.
            Processor planes are ordered by numeric cycle and channel with
            marker names from filenames. The ordered records in
            ``attrs["akoya"]["regions"][region]["channels"]`` retain the original
            biomarker and its one-based acquisition cycle and channel, including
            when the same marker was acquired more than once. Uploader montages keep channels
            within cycles and use complete ImageJ labels, a nearby
            ``channelNames.txt``, or numeric cycle/channel names in that order.
            ``channelNames.txt`` has one nonblank entry per acquisition
            channel across all cycles, encoded as UTF-8 or BOM-marked UTF-16
            and at most 1 MiB. Repeated marker names receive suffixes.

        Coordinate systems:

        - In a single region, shapes use ``table_pixels`` with the identity
          transform or ``global`` with a verified micrometre scale.
        - An independently calibrated image uses ``global`` in micrometres
          when it is the only product and ``image_um`` when there is a
          separate table. An image without calibration uses ``image_pixels``.
          An explicit affine instead maps image pixels into the selected
          table's frame. Uploader downsamples individual tiles before
          assembling a montage, so registration needs an independent mapping.
        - Unrelated Processor and Uploader regions use
          ``<region>_table_pixels`` or ``<region>_table_um`` and
          ``<region>_image_pixels`` or ``<region>_image_um``. CIM scans with
          verified stage positions on the same slide share a micrometre
          ``global`` frame.

        ``attrs`` record the reader and package version. The Akoya version is
        recorded when supplied by ``experiment_file``, or when all regions
        are CIM scans with the same QPI description version.
        Image pixels are not read until computed, and a partial computation
        decodes only the TIFF tiles or strips it overlaps. All files are
        closed on return; caller-owned objects are not modified.

    Warns
    -----
    UserWarning
        If metadata field names were sanitized into table column names.

    Raises
    ------
    TypeError
        If an option has the wrong type.
    ValueError
        If an option has an invalid value, an image option is given without an
        image, ``data_axes`` cannot be mapped to CYX for the image (with a note
        naming it), ``pixel_size_um`` conflicts with the experiment, or the
        calibration overflows a cell position or yields a nonfinite or
        singular physical image transform.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If a path is missing or has the wrong kind.
    ValueError
        A contextual ``ReaderFormatError`` naming the artifact and path if the
        table or image is ambiguous or malformed, for example a
        multi-series TIFF without ``tiff_series``, declared image axes that
        cannot be mapped to CYX, missing or duplicate fields,
        nonfinite intensities or coordinates, several regions, negative,
        non-integral or duplicate cell IDs, or an unsupported experiment
        version.
    """
    validate_calibration(pixel_size_um, image_to_table)
    validate_image_options(data_axes, tiff_series)
    selected = validate_region_options(
        regions, read_tables=read_tables, table_selected=cell_table_file is not None, compensation=compensation
    )
    raster = make_raster_options(scale_factors, chunks)
    layout = discover_layout(
        path,
        cell_table_file=cell_table_file,
        image_file=image_file,
        experiment_file=experiment_file,
        labels=labels,
        labels_to_image=labels_to_image,
        regions=selected,
        read_tables=read_tables,
        compensation=compensation,
        path_policy=path_policy,
    )
    _validate_sources(
        layout,
        raster,
        data_axes=data_axes,
        tiff_series=tiff_series,
        image_to_table=image_to_table,
        calibrate=pixel_size_um is not None or layout.experiment is not None,
    )
    calibration = resolve_calibration(pixel_size_um, layout.experiment)
    images, metadata = _read_images(
        layout, raster, calibration, data_axes=data_axes, tiff_series=tiff_series, registration=image_to_table
    )

    tables: dict[str, AnnData] = {}
    shapes: dict[str, GeoDataFrame] = {}
    masks: dict[str, DataArray | DataTree] = {}
    for source in layout.regions:
        mask_name = "labels" if source.name is None else f"{source.name}_labels"
        mask_target = None
        if source.mask is not None:
            image_name = "images" if source.name is None else f"{source.name}_image"
            mask = read_mask(
                source.mask,
                images[image_name],
                options=raster,
                context=ReaderErrorContext("codex", "segmentation mask"),
            )
            masks[mask_name] = mask
            full_mask = mask["scale0"]["image"] if isinstance(mask, DataTree) else mask
            mask_target = (mask_name, full_mask.dtype)
        if source.cell_table is None:
            continue
        region, table, geometry = _read_region_table(
            source.cell_table,
            region=source.name,
            mask_target=mask_target,
            frame=_table_frame(source.name, multiple=len(layout.regions) > 1, calibrated=calibration is not None),
            calibration=calibration,
        )
        if geometry is not None:
            shapes[region] = geometry
        tables["table" if source.name is None else f"{region}_table"] = table

    result = SpatialData(images=images, labels=masks, tables=tables, shapes=shapes)
    if metadata:
        result.attrs["akoya"] = {"regions": metadata}
    version: str | None = calibration.version if calibration is not None else None
    if version is None and len(metadata) == len(layout.regions):
        versions = {scan.get("description_version") for scan in metadata.values()}
        if len(versions) == 1 and isinstance(declared_version := versions.pop(), str):
            version = declared_version
    set_reader_provenance(result.attrs, reader="codex", format_version=version)
    return result


def _validate_sources(  # noqa: PLR0913 - Independent image and calibration options.
    layout: CodexLayout,
    raster: RasterOptions,
    *,
    data_axes: str | None,
    tiff_series: int | None,
    image_to_table: Affine | None,
    calibrate: bool,
) -> None:
    images = [source.image for source in layout.regions if source.image is not None]
    tiff_options = data_axes is not None or tiff_series is not None
    has_table = any(source.cell_table is not None for source in layout.regions)
    if not images and (image_to_table is not None or tiff_options or raster != make_raster_options()):
        message = "image_to_table, data_axes, tiff_series, and nondefault scale_factors or chunks require an image."
        raise ValueError(message)
    if tiff_options and any(isinstance(source, ScanFiles) for source in images):
        message = "data_axes and tiff_series cannot be used with a CIM QPTIFF."
        raise ValueError(message)
    if image_to_table is not None and (len(layout.regions) != 1 or not has_table):
        message = "image_to_table requires one selected region with both a table and an image."
        raise ValueError(message)
    if calibrate and not has_table:
        message = "pixel_size_um and experiment_file require a CODEX cell table."
        raise ValueError(message)


def _table_frame(region: str | None, *, multiple: bool, calibrated: bool) -> str:
    if multiple:
        return f"{region}_table_um" if calibrated else f"{region}_table_pixels"
    return "global" if calibrated else "table_pixels"


def _read_region_table(
    source: CellTableSource,
    *,
    region: str | None,
    frame: str,
    calibration: Calibration | None,
    mask_target: tuple[str, np.dtype[np.integer]] | None = None,
) -> tuple[str, AnnData, GeoDataFrame | None]:
    cells = read_cell_table(source, expected_region=region)
    table = cells.table
    if source.compensation is not None:
        table.uns["spatialdata_io_compensation"] = source.compensation
    transformation = Identity() if calibration is None else Scale([calibration.pixel_size_um] * 2, axes=("x", "y"))
    target = cells.region
    shapes = None
    if mask_target is None:
        shapes = make_shapes(cells.cell_ids, cells.coordinates, frame=frame, transformation=transformation)
    else:
        target, dtype = mask_target
        if cells.cell_ids.min() == 0 or int(cells.cell_ids.max()) > np.iinfo(dtype).max:
            raise ReaderFormatError(
                context=ReaderErrorContext("codex", "linked cell table"),
                path=source.path,
                found=f"cell IDs outside the positive range of {dtype}",
                expected="nonzero cell IDs representable by the supplied label mask",
            )
        table.obs["region"] = table.obs["region"].cat.rename_categories({cells.region: target})
        table.uns["spatialdata_io_source_region"] = cells.region
    if calibration is not None:
        set_positions(
            table,
            cells.coordinates,
            pixel_size_um=calibration.pixel_size_um,
            calibration=calibration.source,
            frame=frame,
        )

    # The reader builds eager frames; AnnData also annotates backed Dataset2D.
    prepare_anndata_dataframe(
        cast("pd.DataFrame", table.obs),
        context=ReaderErrorContext("codex", "cell observations"),
        allow_float_nulls=True,
    )
    prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=ReaderErrorContext("codex", "cycle features"))
    table = parse_linked_table(
        table,
        TableLinkage((target,), "region", "cell_id"),
        context=ReaderErrorContext("codex", "linked cell table"),
        path=source.path,
        expected_instance_ids=None if shapes is None else {target: shapes.index},
    )
    return cells.region, table, shapes


def _read_images(  # noqa: PLR0913 - Explicit raster axis, series and registration state.
    layout: CodexLayout,
    raster: RasterOptions,
    calibration: Calibration | None,
    *,
    data_axes: str | None,
    tiff_series: int | None,
    registration: Affine | None,
) -> tuple[dict[str, DataArray | DataTree], dict[str, dict[str, object]]]:
    """Construct each regional raster and retain the source identities of measured channels."""
    images: dict[str, DataArray | DataTree] = {}
    multiple = len(layout.regions) > 1
    scans = {source.image.region: source.image for source in layout.regions if isinstance(source.image, ScanFiles)}
    qpi, metadata = (
        _read_cim_scans(
            layout,
            scans,
            raster,
            calibration,
            registration,
        )
        if scans
        else ({}, {})
    )
    for source in layout.regions:
        if source.image is None:
            continue
        name = "images" if source.name is None else f"{source.name}_image"
        if isinstance(source.image, ScanFiles):
            images[name] = qpi[source.image.region]
            continue

        prefix = f"{source.name}_" if multiple else ""
        frame = f"{prefix}image_pixels"
        physical_frame = f"{prefix}image_um" if multiple or source.cell_table is not None else "global"
        if registration is not None:
            frame = _table_frame(source.name, multiple=multiple, calibrated=calibration is not None)
            physical_frame = None
        images[name] = read_image(
            source.image,
            data_axes=data_axes,
            tiff_series=tiff_series,
            raster=raster,
            frame=frame,
            transformation=image_transformation(registration, calibration),
            physical_frame=physical_frame,
        )
        if isinstance(source.image, MarkerSet) and source.name is not None:
            metadata[source.name] = {
                "channels": [
                    {"biomarker": plane.marker, "cycle": plane.cycle, "channel": plane.channel}
                    for plane in source.image.planes
                ]
            }
    return images, metadata


def _read_cim_scans(
    layout: CodexLayout,
    files: dict[str, ScanFiles],
    raster: RasterOptions,
    calibration: Calibration | None,
    registration: Affine | None,
) -> tuple[dict[str, DataArray | DataTree], dict[str, dict[str, object]]]:
    """Apply CODEX table registration, when supplied, to validated Instrument Manager scans."""
    context = ReaderErrorContext("codex", "CIM QPTIFF")
    scans = {region: inspect_scan(source.image, context=context) for region, source in files.items()}
    for region, scan in scans.items():
        source = files[region]
        if scan.generation != "cim_qptiff":
            raise ReaderFormatError(
                context=context,
                path=source.image,
                found=scan.generation,
                expected="a CODEX Instrument Manager scan",
            )
        if source.lower_bit_depth is not None:
            companion = inspect_scan(source.lower_bit_depth, context=context)
            validate_companion(scan, companion, context=context)
    transformations = scan_transformations(scans)
    if len(scans) == 1:
        ((region, scan),) = scans.items()
        multiple = len(layout.regions) > 1
        lone_image = not multiple and layout.regions[0].cell_table is None
        if registration is not None:
            frame = _table_frame(region, multiple=False, calibrated=calibration is not None)
            transformations[region] = {frame: image_transformation(registration, calibration)}
        elif not lone_image or scan.pixel_size_um is None:
            prefix = f"{region}_" if multiple else ""
            frame = f"{prefix}image_pixels" if scan.pixel_size_um is None else f"{prefix}image_um"
            (transformation,) = transformations[region].values()
            transformations[region] = {frame: transformation}

    images = {
        region: read_scan(
            scan,
            options=raster,
            channel_names=None,
            transformations=transformations[region],
            context=context,
        )
        for region, scan in scans.items()
    }
    return images, {region: scan_attributes(scan) for region, scan in scans.items()}
