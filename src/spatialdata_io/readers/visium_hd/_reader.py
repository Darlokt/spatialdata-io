"""Assemble Visium HD products in a calibrated physical slide frame."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from spatialdata import SpatialData

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import (
    invalid_geometry_mask,
    raise_invalid_geometries,
    validate_invalid_geometry_policy,
)
from spatialdata_io.readers._utils.image import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.visium_hd._bins import read_bin_table, read_grid_shape
from spatialdata_io.readers.visium_hd._cells import read_cell_table, read_nucleus_table
from spatialdata_io.readers.visium_hd._geometry import bin_squares, read_boundaries
from spatialdata_io.readers.visium_hd._images import make_image_options, read_images
from spatialdata_io.readers.visium_hd._labels import make_bin_labels
from spatialdata_io.readers.visium_hd._layout import discover_layout
from spatialdata_io.readers.visium_hd._metadata import read_bin_scale_factors, read_identity, read_source_metadata
from spatialdata_io.readers.visium_hd._tables import link_table
from spatialdata_io.readers.visium_hd._transforms import fit_slide_frame

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd
    from anndata import AnnData
    from xarray import DataArray, DataTree

    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.visium_hd._layout import VisiumHDLayout
    from spatialdata_io.readers.visium_hd._transforms import SlideFrame


def _bin_sizes(values: object) -> tuple[int, ...] | Literal["all"]:
    """Validate the public bin selection and own its deterministic order."""
    if isinstance(values, str):
        if values == "all":
            return "all"
        message = "bin_sizes must be 'all' or a sequence of distinct positive even integers."
        raise ValueError(message)

    if not isinstance(values, Sequence):
        message = "bin_sizes must be 'all' or a sequence of distinct positive even integers."
        raise TypeError(message)

    sizes: list[int] = []
    for size in values:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 2:
            message = "bin_sizes must contain positive even integer micrometre sizes."
            raise ValueError(message)
        sizes.append(size)

    if len(set(sizes)) != len(sizes):
        message = "bin_sizes must not contain duplicates."
        raise ValueError(message)

    return tuple(sorted(sizes))


def _select_population(
    table: AnnData, source_ids: pd.Index, geometry: gpd.GeoDataFrame, path: Path
) -> tuple[AnnData, gpd.GeoDataFrame]:
    """Check matrix membership before applying the geometry drop policy."""
    ids = pd.Index(table.obs["instance_id"])
    missing = ids[~ids.isin(source_ids)]
    if len(missing):
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "segmentation"),
            path=path,
            found=f"matrix IDs absent from polygons: {missing[:5].tolist()}",
            expected="every matrix cell in geometry",
        )

    retained = ids.isin(geometry.index)
    if not retained.any():
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "segmentation"),
            path=path,
            found="empty selected population",
            expected="at least one retained cell for table linkage",
        )

    table = table[retained].copy() if not retained.all() else table
    return table, geometry.loc[pd.Index(table.obs["instance_id"])]


def _apply_policy(geometry: gpd.GeoDataFrame, policy: InvalidGeometryPolicy, path: Path) -> gpd.GeoDataFrame:
    """Preserve default source polygons without an unnecessary validity pass."""
    if policy == "keep":
        return geometry

    invalid = invalid_geometry_mask(geometry.geometry.to_numpy())
    if policy == "raise":
        raise_invalid_geometries(
            invalid,
            geometry.index.tolist(),
            context=ReaderErrorContext("visium_hd", "boundaries"),
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

    population = (
        read_cell_table(selected.counts, make_var_names_unique=make_var_names_unique)
        if selected.counts is not None
        else None
    )

    dropped = pd.Index([], dtype=np.int64)
    sources: tuple[tuple[Literal["cell", "nucleus"], ArtifactFile | None], ...] = (
        ("cell", selected.cell_boundaries),
        ("nucleus", selected.nucleus_boundaries),
    )

    for kind, path in sources:
        if path is None:
            continue

        boundaries = read_boundaries(path, frame, kind=kind)
        geometry = _apply_policy(boundaries.shapes, policy, path)
        if kind == "cell":
            dropped = boundaries.shapes.index.difference(geometry.index)
        else:
            geometry = geometry.loc[~geometry.index.isin(dropped)]
        if geometry.empty:
            raise ReaderFormatError(
                context=ReaderErrorContext("visium_hd", kind),
                path=path,
                found="empty boundary selection",
                expected="at least one retained polygon",
            )

        # Matrices define the returned population. Geometry-only nuclei retain source order.
        table = population if kind == "cell" or layout.raw_two_micron_counts is not None else None
        if table is not None:
            table, geometry = _select_population(table, boundaries.shapes.index, geometry, path)

        name = f"{kind}_boundaries"
        shapes[name] = geometry
        if table is None or (kind == "nucleus" and layout.raw_two_micron_counts is None):
            continue

        if kind == "nucleus":
            raw, mapping = layout.raw_two_micron_counts, layout.barcode_mappings
            # Discovery resolves these together when nucleus counts are selected.
            table = read_nucleus_table(
                cast("ArtifactFile", raw), cast("ArtifactFile", mapping), table, boundaries.shapes.index
            )

        if boundaries.centroids is not None:
            table.obsm["spatial"] = boundaries.centroids[boundaries.shapes.index.get_indexer(geometry.index)]
        tables[f"{kind}_table"] = link_table(table, name, geometry.index)

    return shapes, tables


def _validate_options(
    *,
    bin_counts_kind: str,
    cell_counts_kind: str,
    bin_geometry: str,
    include_cell_boundaries: bool | None,
    include_nucleus_boundaries: bool | None,
    include_nucleus_table: bool,
    include_images: bool,
    include_cytassist_image: bool,
    make_var_names_unique: bool,
    fullres_image_file: str | Path | None,
) -> None:
    """Reject incompatible or untyped public option values before discovery."""
    for name, value in (("bin_counts_kind", bin_counts_kind), ("cell_counts_kind", cell_counts_kind)):
        if value not in ("filtered", "raw"):
            message = f"{name} must be 'filtered' or 'raw'."
            raise ValueError(message)

    if bin_geometry not in ("squares", "labels"):
        message = "bin_geometry must be 'squares' or 'labels'."
        raise ValueError(message)

    for boundary_flag in (include_cell_boundaries, include_nucleus_boundaries):
        if boundary_flag is not None and not isinstance(boundary_flag, bool):
            message = "Boundary inclusion options must be booleans or None."
            raise TypeError(message)
    if any(
        not isinstance(value, bool)
        for value in (include_nucleus_table, include_images, include_cytassist_image, make_var_names_unique)
    ):
        message = "Inclusion and feature-name options must be booleans."
        raise TypeError(message)

    if include_nucleus_table and include_nucleus_boundaries is False:
        message = "include_nucleus_table requires nucleus boundaries."
        raise ValueError(message)
    if fullres_image_file is not None and not include_images:
        message = "fullres_image_file requires include_images=True."
        raise ValueError(message)


def visium_hd(
    path: str | Path,
    *,
    bin_sizes: Sequence[int] | Literal["all"] = "all",
    bin_geometry: Literal["squares", "labels"] = "squares",
    bin_counts_kind: Literal["filtered", "raw"] = "filtered",
    include_cell_boundaries: bool | None = None,
    include_nucleus_boundaries: bool | None = None,
    include_nucleus_table: bool = False,
    cell_counts_kind: Literal["filtered", "raw"] = "filtered",
    invalid_geometries: Literal["keep", "drop", "raise"] = "keep",
    include_images: bool = True,
    include_cytassist_image: bool = False,
    fullres_image_file: str | Path | None = None,
    data_axes: Sequence[str] | None = None,
    tiff_series: int | None = None,
    make_var_names_unique: bool = True,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
) -> SpatialData:
    """Read Space Ranger 3.0-4.1 Visium HD output in physical slide micrometres.

    Parameters
    ----------
    path
        Output root containing binned_outputs and feature_slice.h5 (or the
        older hd_feature_slice.h5). Loose prefixed feature files are accepted.
    bin_sizes
        All available sizes by default; a sequence selects exact stored sizes,
        and an empty sequence skips bins. Missing requested sizes raise.
    bin_geometry
        Exact nominal full squares or uint32 grid labels. Partial edge bins
        retain their nominal extent, including the area without barcodes.
    bin_counts_kind, cell_counts_kind
        Independently select filtered or raw matrices, preserving source order.
    include_cell_boundaries, include_nucleus_boundaries
        None reads when present; True requires; False skips. Cell boundaries
        include their count table; nuclei include only geometry by default.
    include_nucleus_table
        Sum raw 2 µm counts over in_nucleus bins in the selected cell population.
        Requires nucleus boundaries; the Scanpy raw read can require several GiB.
    invalid_geometries
        Keep source topology, drop invalid polygons, or raise. Dropping a cell
        also drops its nucleus and table rows; dropping a nucleus affects only
        that nucleus and its table row. Geometry is never repaired.
    include_images
        Read available hires, lowres and full-resolution microscope images.
    include_cytassist_image
        Return a lazy, exactly projectively warped CytAssist image.
    fullres_image_file
        Override full-resolution discovery, relative to the root or absolute.
        Otherwise discover one tissue_image TIFF at root or microscope_image/.
    data_axes, tiff_series
        Explicit full-resolution TIFF axes and series when needed.
    make_var_names_unique
        Make feature names unique in every table; source feature order remains.
    scale_factors
        Generated pyramid factors for the full-resolution image only; empty
        returns one scale. Every factor must be at least two.
    chunks
        Image chunk size or (c, y, x) sizes; labels use y and x sizes.

    Returns
    -------
    spatialdata.SpatialData
        Shapes or labels square_XXXum and square_XXXum_table, optional
        cell_boundaries/cell_table, nucleus_boundaries/nucleus_table, and
        full_image, hires_image, lowres_image, cytassist_image. All use global
        physical micrometres oriented like the microscope image. Vectors and
        float64 obsm['spatial'] are stored in that frame with identity transforms.
        Source pixel positions remain in bin obs. Labels use exact rigid grid
        transforms. Images have (c, y, x) axes, source dtype/channel order and
        a fitted affine, whose maximum residual in µm is recorded in attrs.
        Raster pixels stay lazy; shapes and canonical float32 CSR sparse-array
        tables are eager. Inputs are never mutated; returned lazy sources must
        remain accessible. See the format guide for layouts and memory costs.

    Raises
    ------
    TypeError, ValueError
        Invalid or contradictory public options.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        Missing or wrong-kind artifacts.
    spatialdata_io.readers._utils.errors.ReaderFormatError
        Malformed, ambiguous, unsupported or scientifically inconsistent input,
        including counts that cannot be stored exactly as float32.
    """
    sizes = _bin_sizes(bin_sizes)
    policy = validate_invalid_geometry_policy(invalid_geometries)
    _validate_options(
        bin_counts_kind=bin_counts_kind,
        cell_counts_kind=cell_counts_kind,
        bin_geometry=bin_geometry,
        include_cell_boundaries=include_cell_boundaries,
        include_nucleus_boundaries=include_nucleus_boundaries,
        include_nucleus_table=include_nucleus_table,
        include_images=include_images,
        include_cytassist_image=include_cytassist_image,
        make_var_names_unique=make_var_names_unique,
        fullres_image_file=fullres_image_file,
    )

    image_options = make_image_options(
        scale_factors=scale_factors, chunks=chunks, data_axes=data_axes, tiff_series=tiff_series
    )
    raster = image_options.raster
    if not include_images and (data_axes is not None or tiff_series is not None or raster.scale_factors):
        message = "data_axes, tiff_series and scale_factors require include_images=True."
        raise ValueError(message)

    layout = discover_layout(
        path,
        bin_sizes=sizes,
        bin_counts_kind=bin_counts_kind,
        include_cell_boundaries=include_cell_boundaries,
        include_nucleus_boundaries=include_nucleus_boundaries,
        include_nucleus_table=include_nucleus_table,
        cell_counts_kind=cell_counts_kind,
        include_images=include_images,
        include_cytassist_image=include_cytassist_image,
        fullres_image_file=fullres_image_file,
    )

    identity = read_identity(layout.identity_file)
    metadata = None if layout.feature_slice is None else read_source_metadata(layout.feature_slice)
    scales = read_bin_scale_factors(layout.base_scale_factors, 2)
    shape = read_grid_shape(layout.base_positions) if metadata is None else (metadata.nrows, metadata.ncols)
    frame = fit_slide_frame(layout.base_positions, shape)

    shapes, tables = _segmentations(layout, frame, policy=policy, make_var_names_unique=make_var_names_unique)
    labels: dict[str, DataArray | DataTree] = {}
    for source in layout.bins:
        size = source.size_um
        factor = size // 2
        grid = ((shape[0] + factor - 1) // factor, (shape[1] + factor - 1) // factor)
        table = read_bin_table(source, grid, frame, make_var_names_unique=make_var_names_unique)

        # Shared counts return eager AnnData; the upstream type also includes backed frames.
        obs = cast("pd.DataFrame", table.obs)
        rows = obs["array_row"].to_numpy(dtype=np.int64)
        cols = obs["array_col"].to_numpy(dtype=np.int64)

        ids = pd.Index(np.arange(1, table.n_obs + 1, dtype=np.uint32))
        table.obs["instance_id"] = ids
        name = f"square_{size:03d}um"
        if bin_geometry == "squares":
            shapes[name] = bin_squares(rows, cols, size, frame, ids)
        else:
            labels[name] = make_bin_labels(
                frame, rows.astype(np.uint32), cols.astype(np.uint32), grid, size, options=raster
            )

        tables[f"{name}_table"] = link_table(table, name, ids)

    images: dict[str, DataArray | DataTree] = (
        {} if layout.images is None else read_images(layout.images, frame, scales, metadata, image_options)
    )
    if not shapes and not labels and not images:
        message = "Select at least one available product."
        raise ValueError(message)

    result = SpatialData(images=images, labels=labels, shapes=shapes, tables=tables)
    result.attrs["visium_hd"] = {
        "library_id": identity.library_id,
        "microns_per_pixel": scales.microns_per_pixel,
        "coordinate_units": "micrometres",
        "image_affine_max_residual_um": frame.max_residual_um,
        "image_affine_rms_residual_um": frame.rms_residual_um,
    }
    set_reader_provenance(result.attrs, reader="visium_hd", format_version=identity.format_version)
    return result
