"""Assemble independently selected Stereo-seq products with final exact linkage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from spatialdata import SpatialData
from spatialdata.transformations import Scale, set_transformation

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.geometry import validate_invalid_geometry_policy
from spatialdata_io.readers._utils.image import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS, make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table, prepare_anndata_dataframe
from spatialdata_io.readers.stereoseq._analysis import read_analyses
from spatialdata_io.readers.stereoseq._cells import read_cells
from spatialdata_io.readers.stereoseq._geometry import make_points, make_polygons
from spatialdata_io.readers.stereoseq._layout import discover_layout
from spatialdata_io.readers.stereoseq._metadata import read_image_pitch
from spatialdata_io.readers.stereoseq._rasters import read_image, read_mask
from spatialdata_io.readers.stereoseq._square import read_square

if TYPE_CHECKING:
    import anndata as ad
    import dask.dataframe as dd
    import geopandas as gpd

    from spatialdata_io.readers.stereoseq._cells import CellCounts
    from spatialdata_io.readers.stereoseq._layout import ImageRole
    from spatialdata_io.readers.stereoseq._square import SquareCounts

_ROLES = frozenset({"registered", "mask", "tissue_cut", "distance_mask", "expression"})


def _set_grid_pitch(result: SpatialData, pitch_nm: int) -> None:
    """Map every spatial element's native grid to micrometres without touching its buffers.

    The common pitch can come from analyses read after the raw elements were
    built, so it is attached in one pass over the assembled container.
    """
    scale = Scale([pitch_nm / 1000, pitch_nm / 1000], axes=("x", "y"))
    for elements in (result.images, result.labels, result.points, result.shapes):
        for element in elements.values():
            set_transformation(element, scale, to_coordinate_system="global")


def _sequence[T](values: Sequence[T], name: str) -> tuple[T, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        message = f"{name} must be a sequence, not a scalar or string."
        raise TypeError(message)
    return tuple(values)


def _bin_selection(values: Sequence[int] | Literal["all"]) -> tuple[int, ...] | Literal["all"]:
    if isinstance(values, str) and values == "all":
        return "all"

    bins = _sequence(values, "bin_sizes")
    if any(isinstance(size, bool) or not isinstance(size, int) for size in bins):
        message = "bin_sizes must contain positive integers, or be the scalar 'all'."
        raise TypeError(message)

    if any(size <= 0 for size in bins) or len(set(bins)) != len(bins):
        message = "bin_sizes must contain distinct positive integers."
        raise ValueError(message)
    return tuple(sorted(bins))


def _validate_options(
    *,
    sample_id: str | None,
    cell_bins: Literal["none", "unadjusted", "adjusted", "both"],
    cell_geometry: Literal["centroids", "polygons"],
    roles: tuple[ImageRole, ...],
    analyses: tuple[str | Path, ...],
) -> None:
    """Validate independent product choices before discovery or source reads."""
    if sample_id is not None and (
        not isinstance(sample_id, str)
        or not sample_id
        or sample_id in {".", ".."}
        or any(character in sample_id for character in "/\\\0")
    ):
        message = "sample_id must be a nonempty sample filename component."
        raise ValueError(message)

    if cell_bins not in ("none", "unadjusted", "adjusted", "both"):
        message = "cell_bins must be 'none', 'unadjusted', 'adjusted', or 'both'."
        raise ValueError(message)
    if cell_geometry not in ("centroids", "polygons"):
        message = "cell_geometry must be 'centroids' or 'polygons'."
        raise ValueError(message)

    if any(role not in _ROLES for role in roles) or len(set(roles)) != len(roles):
        message = f"image_roles must be distinct selections from {sorted(_ROLES)}."
        raise ValueError(message)
    if any(not isinstance(file, (str, Path)) for file in analyses):
        message = "analysis_files must contain filesystem paths."
        raise TypeError(message)


def _validate_frames(products: Sequence[SquareCounts | CellCounts], sample: str | None, path: Path) -> None:
    """Reject incompatible grids instead of inventing a crop/registration transform."""
    if not products:
        return

    first = products[0]
    if any(
        product.origin != first.origin
        or product.resolution != first.resolution
        or (product.sample_id is not None and product.sample_id != sample)
        for product in products
    ):
        raise ReaderFormatError(
            context=ReaderErrorContext("stereoseq", "coordinate frame"),
            path=path,
            found="inconsistent source sample, origin or physical pitch",
            expected="matching normalized grids; separately recentered sources require an established transform",
        )


def _link(table: ad.AnnData, target: str, index: pd.Index) -> ad.AnnData:
    """Complete owned annotations once, then validate exact target membership."""
    context = ReaderErrorContext("stereoseq", target)
    obs = cast("pd.DataFrame", table.obs)
    instances = pd.Series(index.to_numpy(), index=obs.index)

    # Imported analysis may already contain linkage columns. Preserve matching
    # values, but never replace a conflicting annotation with an invented link.
    for column, expected in (("region", target), ("instance_id", instances)):
        if column in obs and not obs[column].eq(expected).fillna(value=False).all():
            raise ReaderFormatError(
                context=context,
                found=f"conflicting reserved column {column!r}",
                expected="absent or matching linkage metadata",
            )

    obs["region"] = pd.Categorical.from_codes(np.zeros(len(obs), dtype=np.int8), categories=pd.Index([target]))
    obs["instance_id"] = instances

    # Convert owned annotations only after their final mutation, immediately
    # before the shared AnnData/SpatialData compatibility and linkage boundary.
    prepare_anndata_dataframe(obs, context=context)
    prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=context)
    return parse_linked_table(
        table,
        TableLinkage((target,), "region", "instance_id"),
        context=context,
        expression_layers=tuple(name for name in ("exon", "counts", "log1p") if name in table.layers),
        expected_instance_ids={target: index},
    )


def stereoseq(  # noqa: PLR0913, PLR0915, RUF100 - Independent products retain one explicit transaction outline.
    path: str | Path,
    *,
    sample_id: str | None = None,
    bin_sizes: Sequence[int] | Literal["all"] = (),
    cell_bins: Literal["none", "unadjusted", "adjusted", "both"] = "none",
    cell_geometry: Literal["centroids", "polygons"] = "centroids",
    invalid_geometries: Literal["keep", "drop", "raise"] = "keep",
    square_counts_file: str | Path | None = None,
    cell_counts_file: str | Path | None = None,
    adjusted_cell_counts_file: str | Path | None = None,
    analysis_files: Sequence[str | Path] = (),
    image_roles: Sequence[Literal["registered", "mask", "tissue_cut", "distance_mask", "expression"]] = (),
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
) -> SpatialData:
    """Read independent SAW transcriptomic counts, clustering analysis and images.

    Supports the documented compact numbered workflow, SAW 7.0/7.1 numbered
    layouts, SAW 8.0/8.1/8.2 ``outs`` layouts, and explicit flat count exports.
    Directory family and GEF schema do not identify a SAW release; provenance
    records only reader/package versions. See the Stereo-seq format guide for
    directory trees, tested storage profiles, and migration examples.

    Parameters
    ----------
    path
        Dataset root, task root containing ``outs``, or direct ``outs`` path.
    sample_id
        Source sample selector. Required when multiple samples are present.
        Dots in sample names are preserved; this is not an output-name prefix.
    bin_sizes
        Distinct positive stored bin sizes, sorted numerically, or ``"all"``.
        Empty selects no square counts. Missing bins fail; none are derived.
    cell_bins
        Select no cells, unadjusted cells, adjusted cells, or both populations.
        Raw cells do not require square counts, analysis, or images.
    cell_geometry
        Source centroids or literal source polygons. Centroid mode never
        reads borders. Polygon mode applies the topology policy without repair.
        Empty selected populations cannot be linked and fail.
    invalid_geometries
        Preserve invalid source topology, drop affected cells and matching
        count rows/layers, or raise. Raw adjusted and unadjusted populations
        are independent; analyses and masks remain unchanged. Paired-padding
        errors always fail, and centroid mode does not inspect borders.
    square_counts_file
        Explicit square GEF. By default select the unique tissue/full GEF;
        ambiguous sources fail. Whole-chip raw counts require this override.
    cell_counts_file, adjusted_cell_counts_file
        Explicit sources for their selected cell populations. Overrides do not
        select products. Relative paths resolve under ``path``; absolute caller
        paths are allowed. Flat counts require explicit paths.
    analysis_files
        Exact native SAW H5AD paths. Preserve analysis values, source order,
        annotations, embeddings and raw. A matching selected GEF validates
        filtered membership and coordinates by source keys. Analysis-only
        requires native spatial metadata and constructs its own point target.
    image_roles
        Registered microscopy, binary masks, tissue-cut images, expanded masks,
        or ``bin1_img.tif`` expression intensity. All stains for each selected
        role are included. Masks, tissue cuts and expanded masks become labels.
    scale_factors
        Integer factors of at least two, one per generated pyramid level;
        empty for a single scale. The same factors apply to every selected
        image and mask.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes; by default
        one channel per chunk and square 1024-pixel tiles. Masks use the
        ``y, x`` sizes. Pixels remain lazy after return.

    Returns
    -------
    SpatialData
        ``bin<N>_genes`` points and ``bin<N>_table`` hold source-ordered features
        and first-occurrence coordinates, including zero-count positions and
        zero-occurrence features. Cells use stable source IDs in
        ``cells_centroids`` points or ``cells_polygons`` shapes and
        ``cells_table``; adjusted products use the ``adjusted_cells`` prefix.
        Raw expression and optional ``exon`` layers are eager canonical uint64
        CSR sparse arrays. Source summaries remain annotations, not replacements
        for exact expression. No NNZ-sized alternate traversal is copied to uns.

        Analysis products use ``<population>_analysis[_r<resolution>]`` plus
        ``_points``/``_table``; decimal points in clustering resolution become
        underscores. Historical unknown resolution is omitted. Images and
        labels use ``<stain>_<role>`` (distance masks append their pixel
        distance), or the role alone without a stain. ``bin1_img_tissue_cut.tif``
        becomes ``expression_tissue_cut`` to distinguish its mask from
        microscopy. Registered and expression images have CYX axes with source
        dtype and channel order; RGB(A) samples use r/g/b/a channel names.
        Masks are YX labels with their source boolean or integer dtype and
        values; generated levels downsample by nearest neighbour.

        All elements map to ``global`` in micrometres using the common source
        pitch in nm divided by 1000. Native values and table embeddings remain
        in the matching normalized expression grid (XY in tables/points).
        Image-only reads use associated GEF metadata when available, otherwise
        retain the source grid. Stored square coordinates are
        already in that grid and are not multiplied by bin size. Common chip
        origins are retained as metadata, never added to only one element.
        Conflicting origins/pitches fail. Physical pitch is recorded in nm.
        Points wrap eager coordinates needed for counts; GEF ingestion itself
        is eager with bounded input slices and O(NNZ) working/output storage.
        Files close on success/failure. No caller-owned data are mutated.

    Raises
    ------
    TypeError, ValueError
        Invalid options, no selected products, unsupported axes (including
        multichannel masks) or invalid data.
        Format errors use the contextual ``ReaderFormatError`` subclass:
        ambiguous layouts, malformed schemas, overflow, identity/coordinate
        disagreement, invalid polygons, non-integer masks, or missing selected
        products.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        Missing selected artifacts or incorrect filesystem kinds.
    """
    policy = validate_invalid_geometry_policy(invalid_geometries)
    bins = _bin_selection(bin_sizes)
    roles = _sequence(image_roles, "image_roles")
    analyses = _sequence(analysis_files, "analysis_files")
    _validate_options(
        sample_id=sample_id,
        cell_bins=cell_bins,
        cell_geometry=cell_geometry,
        roles=roles,
        analyses=analyses,
    )
    raster = make_raster_options(scale_factors, chunks)
    if not bins and cell_bins == "none" and not roles and not analyses:
        message = "Select products, e.g. stereoseq(path, bin_sizes=(50,)) or stereoseq(path, cell_bins='unadjusted')."
        raise ValueError(message)
    if (
        (square_counts_file is not None and not bins)
        or (cell_counts_file is not None and cell_bins not in ("unadjusted", "both"))
        or (adjusted_cell_counts_file is not None and cell_bins not in ("adjusted", "both"))
    ):
        message = "Count-source overrides require selection of their corresponding product."
        raise ValueError(message)

    layout = discover_layout(
        path,
        sample_id=sample_id,
        bins=bins,
        cell_bins=cell_bins,
        square_counts_file=square_counts_file,
        cell_counts_file=cell_counts_file,
        adjusted_cell_counts_file=adjusted_cell_counts_file,
        analysis_files=analyses,
        image_roles=roles,
    )
    square_counts = tuple(product for source in layout.squares for product in read_square(source))
    cell_counts = tuple(read_cells(source, read_borders=cell_geometry == "polygons") for source in layout.cells)
    products: tuple[SquareCounts | CellCounts, ...] = (*square_counts, *cell_counts)
    _validate_frames(products, layout.sample_id, layout.root)

    # Raw populations establish independent targets before any analysis is
    # matched against them; no analysis row order determines a raw identity.
    points: dict[str, dd.DataFrame] = {}
    shapes: dict[str, gpd.GeoDataFrame] = {}
    tables: dict[str, ad.AnnData] = {}
    counts_by_population: dict[str, SquareCounts | CellCounts] = {}

    for square in square_counts:
        name = f"bin{square.bin_size}"
        target = f"{name}_genes"
        square_index = pd.RangeIndex(len(square.coordinates))
        points[target] = make_points(square.coordinates, square_index)
        counts_by_population[name] = square
        tables[f"{name}_table"] = _link(square.table, target, square_index)

    for cell_source, cells in zip(layout.cells, cell_counts, strict=True):
        name = "cells" if cell_source.kind == "unadjusted" else "adjusted_cells"
        target = f"{name}_{cell_geometry}"
        index = pd.Index(cells.cell_ids)
        if cells.borders is None:
            points[target] = make_points(cells.coordinates, index)
        else:
            shapes[target] = make_polygons(
                cells.coordinates, cells.borders, index, path=cell_source.file, policy=policy
            )
            # Dropped contours remove their count rows; analyses still match the full raw population.
            index = shapes[target].index
        table = (
            cells.table if len(index) == cells.table.n_obs else cells.table[pd.Index(cells.cell_ids).isin(index)].copy()
        )
        counts_by_population[name] = cells
        tables[f"{name}_table"] = _link(table, target, index)

    analyses, pitch = read_analyses(
        layout.analyses,
        counts=counts_by_population,
        resolution=next((product.resolution for product in products), None),
    )
    for source, analysis in zip(layout.analyses, analyses, strict=True):
        name = f"{analysis.name}_table"
        target = f"{analysis.name}_points"
        points[target] = make_points(analysis.coordinates, analysis.instance_ids)
        try:
            tables[name] = _link(analysis.table, target, analysis.instance_ids)
        except (TypeError, ValueError) as error:
            add_reader_context(error, context=ReaderErrorContext("stereoseq", "analysis table"), path=source.file)
            raise

    result = SpatialData(
        images={source.name: read_image(source, raster=raster) for source in layout.images if not source.is_mask},
        labels={source.name: read_mask(source, raster=raster) for source in layout.images if source.is_mask},
        points=points,
        shapes=shapes,
        tables=tables,
    )
    # Image-only selections take the pitch from associated GEF companions, if any.
    if pitch is None:
        pitch = read_image_pitch(layout.calibration, sample_id=layout.sample_id)
    if pitch is not None:
        _set_grid_pitch(result, pitch)
    set_reader_provenance(result.attrs, reader="stereoseq")
    return result
