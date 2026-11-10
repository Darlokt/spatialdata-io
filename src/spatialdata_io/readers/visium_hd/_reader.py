"""Assemble Visium HD products in a calibrated physical slide frame.

One read validates every public option, discovers the selected artifacts and
their shared Space Ranger identity, fits the slide registration from the 2 µm
positions, builds segmentation shapes and tables (cells before nuclei, so that
dropped cells also drop their nuclei), then bins and images, links each table
once and records provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from spatialdata import SpatialData

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import (
    invalid_geometry_mask,
    raise_invalid_geometries,
    validate_invalid_geometry_policy,
)
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.visium_hd._cells import read_cell_table, read_nucleus_table
from spatialdata_io.readers.visium_hd._geometry import bin_squares, read_boundaries
from spatialdata_io.readers.visium_hd._images import make_image_options, read_images
from spatialdata_io.readers.visium_hd._labels import make_bin_labels
from spatialdata_io.readers.visium_hd._layout import discover_layout
from spatialdata_io.readers.visium_hd._metadata import read_bin_scale_factors, read_source_metadata
from spatialdata_io.readers.visium_hd._positions import fit_slide_frame, read_grid_shape
from spatialdata_io.readers.visium_hd._tables import link_table, read_bin_table
from spatialdata_io.readers.visium_hd._transforms import SPOT_PITCH_UM

if TYPE_CHECKING:
    import geopandas as gpd
    from anndata import AnnData
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.visium_hd._images import ImageOptions
    from spatialdata_io.readers.visium_hd._layout import VisiumHDLayout
    from spatialdata_io.readers.visium_hd._transforms import SlideFrame


@dataclass(frozen=True, slots=True)
class _Options:
    """Owned normalized options that discovery does not consume directly."""

    bin_sizes: tuple[int, ...] | Literal["all"]
    invalid_geometries: InvalidGeometryPolicy
    images: ImageOptions


def _bin_sizes(values: object) -> tuple[int, ...] | Literal["all"]:
    """Validate the public bin selection and own its deterministic order."""
    if isinstance(values, str):
        if values == "all":
            return "all"
        message = f"bin_sizes must be 'all' or a sequence of distinct positive even integers; found {values!r}."
        raise ValueError(message)
    if not isinstance(values, Sequence):
        message = f"bin_sizes must be 'all' or a sequence of integers; found {type(values).__name__}."
        raise TypeError(message)

    sizes: list[int] = []
    for size in values:
        if isinstance(size, bool) or not isinstance(size, int):
            message = f"bin_sizes must contain integers; found {type(size).__name__}."
            raise TypeError(message)
        if size <= 0 or size % 2:
            message = f"bin_sizes must contain positive even micrometre sizes; found {size}."
            raise ValueError(message)
        sizes.append(size)

    if len(set(sizes)) != len(sizes):
        message = "bin_sizes must not contain duplicates."
        raise ValueError(message)

    return tuple(sorted(sizes))


def _choice(value: object, *, parameter: str, choices: tuple[str, ...]) -> None:
    """Reject a non-string (``TypeError``) or an unsupported string (``ValueError``)."""
    if not isinstance(value, str):
        message = f"{parameter} must be one of {choices}; found {type(value).__name__}."
        raise TypeError(message)
    if value not in choices:
        message = f"{parameter} must be one of {choices}; found {value!r}."
        raise ValueError(message)


def _flag(value: object, *, parameter: str, optional: bool = False) -> None:
    """Reject a value that is not a boolean (or ``None`` where that means 'when present')."""
    if not isinstance(value, bool) and not (optional and value is None):
        message = f"{parameter} must be a boolean{' or None' if optional else ''}; found {type(value).__name__}."
        raise TypeError(message)


def _validate_options(  # noqa: PLR0913, RUF100 - Validate every public option together, in signature order.
    *,
    bin_sizes: object,
    bin_geometry: object,
    bin_counts_kind: object,
    include_cell_boundaries: object,
    include_nucleus_boundaries: object,
    include_nucleus_table: object,
    cell_counts_kind: object,
    make_var_names_unique: object,
    invalid_geometries: object,
    include_images: object,
    include_cytassist_image: object,
    fullres_image_file: object,
    data_axes: str | None,
    tiff_series: int | None,
    scale_factors: Sequence[int],
    chunks: int | tuple[int, int, int],
) -> _Options:
    """Validate each public option in signature order, then their combinations, before any I/O.

    ``path_policy`` comes last and is validated by discovery before it touches
    the filesystem.
    """
    sizes = _bin_sizes(bin_sizes)
    _choice(bin_geometry, parameter="bin_geometry", choices=("squares", "labels"))
    _choice(bin_counts_kind, parameter="bin_counts_kind", choices=("filtered", "raw"))
    _flag(include_cell_boundaries, parameter="include_cell_boundaries", optional=True)
    _flag(include_nucleus_boundaries, parameter="include_nucleus_boundaries", optional=True)
    _flag(include_nucleus_table, parameter="include_nucleus_table")
    _choice(cell_counts_kind, parameter="cell_counts_kind", choices=("filtered", "raw"))
    _flag(make_var_names_unique, parameter="make_var_names_unique")
    policy = validate_invalid_geometry_policy(invalid_geometries)
    _flag(include_images, parameter="include_images")
    _flag(include_cytassist_image, parameter="include_cytassist_image")
    if fullres_image_file is not None and not isinstance(fullres_image_file, str | Path):
        message = f"fullres_image_file must be a string, a path or None; found {type(fullres_image_file).__name__}."
        raise TypeError(message)
    images = make_image_options(
        data_axes=data_axes, tiff_series=tiff_series, scale_factors=scale_factors, chunks=chunks
    )

    if include_nucleus_table and include_nucleus_boundaries is False:
        message = "include_nucleus_table requires nucleus boundaries; do not pass include_nucleus_boundaries=False."
        raise ValueError(message)
    if not include_images and (
        fullres_image_file is not None
        or data_axes is not None
        or tiff_series is not None
        or images.raster.scale_factors
    ):
        message = "fullres_image_file, data_axes, tiff_series and scale_factors require include_images=True."
        raise ValueError(message)

    return _Options(sizes, policy, images)


def _select_population(
    table: AnnData, source_ids: pd.Index, geometry: gpd.GeoDataFrame, path: Path, *, version: str | None
) -> tuple[AnnData, gpd.GeoDataFrame]:
    """Check matrix membership before applying the geometry drop policy."""
    context = ReaderErrorContext("visium_hd", "segmentation", version)
    ids = pd.Index(table.obs["instance_id"])
    missing = ids[~ids.isin(source_ids)]
    if len(missing):
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"matrix IDs absent from polygons: {missing[:5].tolist()}",
            expected="every matrix cell in geometry",
        )

    retained = ids.isin(geometry.index)
    if not retained.any():
        raise ReaderFormatError(
            context=context,
            path=path,
            found="empty selected population",
            expected="at least one retained cell for table linkage",
        )

    table = table[retained].copy() if not retained.all() else table
    return table, geometry.loc[pd.Index(table.obs["instance_id"])]


def _apply_policy(
    geometry: gpd.GeoDataFrame, policy: InvalidGeometryPolicy, path: Path, *, kind: str, version: str | None
) -> gpd.GeoDataFrame:
    """Preserve default source polygons without an unnecessary validity pass."""
    if policy == "keep":
        return geometry

    invalid = invalid_geometry_mask(geometry.geometry.to_numpy())
    if policy == "raise":
        raise_invalid_geometries(
            invalid,
            geometry.index.tolist(),
            context=ReaderErrorContext("visium_hd", f"{kind} boundaries", version),
            path=path,
        )
        return geometry

    return geometry.loc[~invalid]


def _segmentations(
    layout: VisiumHDLayout,
    frame: SlideFrame,
    *,
    policy: InvalidGeometryPolicy,
    make_var_names_unique: bool,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, AnnData]]:
    """Apply cell drops before nucleus selection, then link each final table."""
    shapes: dict[str, gpd.GeoDataFrame] = {}
    tables: dict[str, AnnData] = {}
    selected = layout.segmentation
    if selected is None:
        return shapes, tables

    version = layout.version
    population = (
        None
        if selected.counts is None
        else read_cell_table(selected.counts, make_var_names_unique=make_var_names_unique, version=version)
    )

    dropped = pd.Index([], dtype=np.int64)
    sources: tuple[tuple[Literal["cell", "nucleus"], ArtifactFile | None], ...] = (
        ("cell", selected.cell_boundaries),
        ("nucleus", selected.nucleus_boundaries),
    )

    for kind, path in sources:
        if path is None:
            continue

        boundaries = read_boundaries(path, frame, kind=kind, version=version)
        geometry = _apply_policy(boundaries.shapes, policy, path, kind=kind, version=version)
        if kind == "cell":
            dropped = boundaries.shapes.index.difference(geometry.index)
        else:
            geometry = geometry.loc[~geometry.index.isin(dropped)]
        if geometry.empty:
            raise ReaderFormatError(
                context=ReaderErrorContext("visium_hd", f"{kind} boundaries", version),
                path=path,
                found="empty boundary selection",
                expected="at least one retained polygon",
            )

        # Matrices define the returned population. Geometry-only nuclei retain source order.
        table = population if kind == "cell" or layout.nucleus_counts is not None else None
        if table is not None:
            table, geometry = _select_population(table, boundaries.shapes.index, geometry, path, version=version)

        name = f"{kind}_boundaries"
        shapes[name] = geometry
        if table is None or selected.counts is None:
            continue

        counts_path = selected.counts
        if kind == "nucleus" and (source := layout.nucleus_counts) is not None:
            counts_path = source.raw_two_micron_counts
            table = read_nucleus_table(
                source.raw_two_micron_counts,
                source.barcode_mappings,
                table,
                boundaries.shapes.index,
                version=version,
            )

        if boundaries.centroids is not None:
            table.obsm["spatial"] = boundaries.centroids[boundaries.shapes.index.get_indexer(geometry.index)]
        tables[f"{kind}_table"] = link_table(
            table, name, geometry.index, position=f"vendor {kind} centroid", counts_path=counts_path, version=version
        )

    return shapes, tables


def visium_hd(  # noqa: PLR0913, RUF100 - Explicit supported products and format choices.
    path: str | Path,
    *,
    bin_sizes: Sequence[int] | Literal["all"] = "all",
    bin_geometry: Literal["squares", "labels"] = "squares",
    bin_counts_kind: Literal["filtered", "raw"] = "filtered",
    include_cell_boundaries: bool | None = None,
    include_nucleus_boundaries: bool | None = None,
    include_nucleus_table: bool = False,
    cell_counts_kind: Literal["filtered", "raw"] = "filtered",
    make_var_names_unique: bool = True,
    invalid_geometries: Literal["keep", "drop", "raise"] = "keep",
    include_images: bool = True,
    include_cytassist_image: bool = False,
    fullres_image_file: str | Path | None = None,
    data_axes: str | None = None,
    tiff_series: int | None = None,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read one Space Ranger Visium HD output directory in slide micrometres.

    Space Ranger 3.x writes binned outputs; 4.x adds segmentation::

        outs/
        ├── feature_slice.h5                   # 3.0: hd_feature_slice.h5
        ├── barcode_mappings.parquet           # 4.x; or inside segmented_outputs/
        ├── binned_outputs/
        │   ├── square_002um/                  # always required: calibrates every read
        │   │   ├── filtered_feature_bc_matrix.h5
        │   │   ├── raw_feature_bc_matrix.h5
        │   │   └── spatial/
        │   │       ├── tissue_positions.parquet
        │   │       ├── scalefactors_json.json
        │   │       ├── tissue_hires_image.png  # optional
        │   │       ├── tissue_lowres_image.png # optional
        │   │       └── cytassist_image.tiff    # optional
        │   └── square_008um/, square_016um/, ...  # same structure, any stored even size
        ├── segmented_outputs/                 # 4.x
        │   ├── filtered_feature_cell_matrix.h5
        │   ├── raw_feature_cell_matrix.h5
        │   ├── cell_segmentations.geojson
        │   └── nucleus_segmentations.geojson
        ├── spatial/                           # optional; its images take precedence
        └── tissue_image.tif                   # optional full-resolution image (or microscope_image/)

    10x sample downloads may prefix the root files with a dataset name. The
    layout is recognized from the artifacts present, not from the Space
    Ranger version; the reader is tested with Space Ranger 3.0, 3.1, 4.0 and
    4.1. Each consumed count matrix and feature slice reports its version as
    ``software_version``; those spelled ``[spaceranger-]X.Y.Z`` must agree, and
    at most one distinct ``library_ids`` entry may occur. The format guide
    describes discovery in detail.

    Parameters
    ----------
    path
        Space Ranger ``outs`` directory.
    bin_sizes
        Bin sizes in micrometres to return, for example ``(8, 16)``; ``"all"``
        (default) returns every stored size and an empty sequence none. A
        requested size that is not stored raises.
    bin_geometry
        ``"squares"`` returns each bin size as nominal square polygons;
        ``"labels"`` as one ``uint32`` label image with one pixel per bin,
        which is far smaller for the 2 µm grid.
    bin_counts_kind
        ``"filtered"`` (in-tissue bins) or ``"raw"`` (every bin) count matrices.
    include_cell_boundaries, include_nucleus_boundaries
        ``None`` reads the segmentation when present, ``True`` requires it,
        ``False`` skips it. Cell boundaries come with ``cell_table``; nucleus
        boundaries with a table only through ``include_nucleus_table``.
    include_nucleus_table
        Add ``nucleus_table``, summing raw 2 µm counts over each nucleus.
        Requires nucleus boundaries, so ``include_nucleus_boundaries`` must not
        be ``False``. Reading the raw 2 µm matrix can take several GiB.
    cell_counts_kind
        ``"filtered"`` or ``"raw"`` cell matrix; it also selects the nucleus
        table's population.
    make_var_names_unique
        Make duplicate feature names unique by appending ``-1``, ``-2``, ...
        in every table.
    invalid_geometries
        ``"keep"`` returns source polygons unchanged, ``"drop"`` removes
        invalid ones (an invalid cell also removes its nucleus and both table
        rows; an invalid nucleus only its own polygon and row), ``"raise"``
        fails. Geometry is never repaired.
    include_images
        Return the available hires, lowres and full-resolution microscope
        images.
    include_cytassist_image
        Return the CytAssist image, lazily resampled into ``global``. Requires
        the feature slice with a non-null CytAssist registration.
    fullres_image_file
        Full-resolution microscope image to read instead of the discovered
        one: TIFF-family (recognized by content), PNG or JPEG. Relative paths
        resolve against ``path``; absolute paths may point elsewhere. Requires
        ``include_images``.
    data_axes
        Source axes of a full-resolution TIFF as one string of distinct letters
        including ``X`` and ``Y``, case-insensitive, for example ``"CYX"``;
        ``S`` (samples) counts as a channel axis. Defaults to the axes the file
        declares.
    tiff_series
        Zero-based series of a full-resolution TIFF. Required when the file
        has several series.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        full-resolution image; empty for a single scale. Other images stay
        single-scale.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes, applied to
        every image; labels use the ``y, x`` sizes.
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
    SpatialData
        The selected elements and their linked tables, all in the ``global``
        coordinate system:

        ``shapes`` or ``labels["square_XXXum"]`` and ``tables["square_XXXum_table"]``
            One element and table per selected bin size (``square_008um``).
            Squares have the nominal side length, also at the partial last
            row and column; labels hold the table-order ``instance_id``
            (1..n) of each bin and 0 where the matrix has no bin. Table rows
            follow the matrix order; ``obs`` has ``in_tissue``, ``array_row``,
            ``array_col`` and the source pixel positions ``pxl_row_in_fullres``
            and ``pxl_col_in_fullres``; ``obsm["spatial"]`` holds the nominal
            bin centres.
        ``shapes["cell_boundaries"]``, ``tables["cell_table"]``
            Cell polygons indexed by uint32 ``cell_id`` and the selected cell
            matrix, linked by uint32 ``instance_id``; ``obsm["spatial"]`` holds the
            vendor centroids (Space Ranger 4.1 and later).
        ``shapes["nucleus_boundaries"]``, ``tables["nucleus_table"]``
            Nucleus polygons and, when requested, the summed raw 2 µm counts
            of the selected cells' nuclei, with vendor nucleus centroids.
            Shape indexes and table ``instance_id`` use the same uint32 cell IDs.
        ``images["full_image"]``, ``images["hires_image"]``, ``images["lowres_image"]``
            Lazy CYX microscope images with the source dtype and channel order;
            only the full-resolution image gets ``scale_factors`` levels.
        ``images["cytassist_image"]``
            Lazy CYX CytAssist image, bilinearly resampled at its own physical
            resolution in the source dtype.

        Tables: ``X`` is a canonical CSR sparse array, float32 through 2**24,
        else float64 through 2**53; already decoded floating values are not
        widened. Duplicate HDF5 entries fail. ``var`` keeps ``gene_ids``,
        ``feature_types`` and ``genome`` in matrix order; ``region`` and
        ``instance_id`` link each row to its element;
        ``uns["spatialdata_io_positions"]`` declares ``obsm["spatial"]``
        (float64 ``x, y`` micrometres), which is absent when the source has no
        centroids. Polygon centroids are never substituted.

        Image channels are named ``r, g, b`` (``r, g, b, a`` with declared
        alpha) only when the file declares RGB: TIFF
        ``PhotometricInterpretation`` RGB or JPEG-compressed YCbCr, PNG colour
        types 2, 3 and 6, and three-component JPEG. Otherwise OME channel
        names, or IDs, name the channels (``<name or ID>:sample:<index>`` for
        an OME channel with several samples); other images use positions. A
        four-channel fluorescence image is never labelled RGB(A).

        Coordinates: ``global`` is micrometres on the manufactured 2 µm barcode
        grid, rigidly oriented like the microscope image, with its origin at
        the fitted position of full-resolution pixel ``(0, 0)``. Shapes and
        ``obsm["spatial"]`` are stored in it with identity transforms;
        segmentation vertices are reprojected exactly from image pixels.
        Labels carry an exact scale and rigid transform. Microscope images
        carry a least-squares affine whose maximum and RMS residual in µm
        are stored in ``attrs["visium_hd"]`` and in each image's ``attrs``.
        There is no pixel coordinate system.

        ``attrs`` record the reader, the package version, the Space Ranger
        version as format version (``4.0.1`` for ``spaceranger-4.0.1``; none
        if no source reports an ``X.Y.Z`` version), and ``attrs["visium_hd"]``
        the library ID (or ``None``) and Space Ranger's nominal
        ``microns_per_pixel``. Image pixels are not read until computed;
        shapes and tables are eager. No source file is modified and lazy
        images keep reading their source files.

    Warns
    -----
    UserWarning
        Once if the detected Space Ranger release is not one of the tested
        releases 3.0, 3.1, 4.0 and 4.1; the read proceeds.

    Raises
    ------
    TypeError
        If an option has the wrong type.
    ValueError
        If an option has an invalid value or contradicts another option,
        ``data_axes`` cannot be mapped to CYX for the image (with a note naming
        it), no product is selected, or ``data_axes``/``tiff_series`` are given
        for a PNG or JPEG image.
    IndexError
        If ``tiff_series`` is out of range.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If a required or requested artifact is missing or has the wrong kind.
    ValueError
        A contextual ``ReaderFormatError`` naming the artifact, path and
        detected Space Ranger version if an artifact is ambiguous, malformed,
        unsupported or inconsistent with another, for example sources reporting
        different Space Ranger versions or library IDs, positions that disagree
        with the 2 µm registration, negative or fractional counts, counts
        whose nucleus sums cannot stay exact, a multi-series TIFF without
        ``tiff_series``, or image axes declared by a file that cannot be
        mapped to CYX.
    """
    options = _validate_options(
        bin_sizes=bin_sizes,
        bin_geometry=bin_geometry,
        bin_counts_kind=bin_counts_kind,
        include_cell_boundaries=include_cell_boundaries,
        include_nucleus_boundaries=include_nucleus_boundaries,
        include_nucleus_table=include_nucleus_table,
        cell_counts_kind=cell_counts_kind,
        make_var_names_unique=make_var_names_unique,
        invalid_geometries=invalid_geometries,
        include_images=include_images,
        include_cytassist_image=include_cytassist_image,
        fullres_image_file=fullres_image_file,
        data_axes=data_axes,
        tiff_series=tiff_series,
        scale_factors=scale_factors,
        chunks=chunks,
    )

    layout = discover_layout(
        path,
        bin_sizes=options.bin_sizes,
        bin_counts_kind=bin_counts_kind,
        include_cell_boundaries=include_cell_boundaries,
        include_nucleus_boundaries=include_nucleus_boundaries,
        include_nucleus_table=include_nucleus_table,
        cell_counts_kind=cell_counts_kind,
        include_images=include_images,
        include_cytassist_image=include_cytassist_image,
        fullres_image_file=fullres_image_file,
        path_policy=path_policy,
    )
    version = layout.version

    metadata = None if layout.feature_slice is None else read_source_metadata(layout.feature_slice, version=version)
    scales = read_bin_scale_factors(layout.base_scale_factors, 2, version=version)
    for source in layout.bins:
        if source.size_um != SPOT_PITCH_UM or source.scale_factors != layout.base_scale_factors:
            read_bin_scale_factors(source.scale_factors, source.size_um, version=version)

    shape = (
        read_grid_shape(layout.base_positions, version=version)
        if metadata is None
        else (metadata.nrows, metadata.ncols)
    )
    frame = fit_slide_frame(layout.base_positions, shape, version=version)

    shapes, tables = _segmentations(
        layout, frame, policy=options.invalid_geometries, make_var_names_unique=make_var_names_unique
    )
    labels: dict[str, DataArray | DataTree] = {}
    for source in layout.bins:
        size = source.size_um
        factor = size // 2
        grid = ((shape[0] + factor - 1) // factor, (shape[1] + factor - 1) // factor)
        table = read_bin_table(source, grid, frame, make_var_names_unique=make_var_names_unique, version=version)

        # Shared counts return eager AnnData; the upstream type also includes backed frames.
        obs = cast("pd.DataFrame", table.obs)
        rows = obs["array_row"].to_numpy(dtype=np.uint32)
        cols = obs["array_col"].to_numpy(dtype=np.uint32)

        ids = pd.Index(np.arange(1, table.n_obs + 1, dtype=np.uint32))
        table.obs["instance_id"] = ids
        name = f"square_{size:03d}um"
        if bin_geometry == "squares":
            shapes[name] = bin_squares(rows, cols, size, frame, ids)
        else:
            labels[name] = make_bin_labels(frame, rows, cols, grid, size, options=options.images.raster)

        tables[f"{name}_table"] = link_table(
            table, name, ids, position="nominal bin centre", counts_path=source.counts, version=version
        )

    images: dict[str, DataArray | DataTree] = (
        {}
        if layout.images is None
        else read_images(layout.images, frame, scales, metadata, options.images, version=version)
    )
    if not shapes and not labels and not images:
        message = "Select at least one available product."
        raise ValueError(message)

    result = SpatialData(images=images, labels=labels, shapes=shapes, tables=tables)
    result.attrs["visium_hd"] = {
        "library_id": layout.identity.library_id,
        "microns_per_pixel": scales.microns_per_pixel,
        "coordinate_units": "micrometres",
        "image_affine_max_residual_um": frame.max_residual_um,
        "image_affine_rms_residual_um": frame.rms_residual_um,
    }
    set_reader_provenance(result.attrs, reader="visium_hd", format_version=version)
    return result
