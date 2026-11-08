"""Public Stereo-seq reader: select products, read them, put them in one frame and link tables."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import pandas as pd
from spatialdata import SpatialData
from spatialdata.transformations import Identity, Scale

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import validate_invalid_geometry_policy
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.stereoseq._analysis import read_analyses
from spatialdata_io.readers.stereoseq._cells import read_cells
from spatialdata_io.readers.stereoseq._geometry import make_points, make_polygons
from spatialdata_io.readers.stereoseq._layout import discover_layout
from spatialdata_io.readers.stereoseq._metadata import read_image_pitch
from spatialdata_io.readers.stereoseq._rasters import read_image, read_mask
from spatialdata_io.readers.stereoseq._square import read_square
from spatialdata_io.readers.stereoseq._tables import finalize_positions, link_table, select_cells

if TYPE_CHECKING:
    from collections.abc import Mapping

    import anndata as ad
    import dask.dataframe as dd
    from spatialdata.transformations import BaseTransformation

    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers._utils.table import SparseDuplicateSummary
    from spatialdata_io.readers.stereoseq._analysis import AnalysisResult
    from spatialdata_io.readers.stereoseq._cells import CellCounts
    from spatialdata_io.readers.stereoseq._layout import AnalysisSource, CellSource, ImageRole
    from spatialdata_io.readers.stereoseq._square import SquareCounts

_CELL_BINS: tuple[Literal["none", "unadjusted", "adjusted", "both"], ...] = ("none", "unadjusted", "adjusted", "both")
_CELL_GEOMETRIES: tuple[Literal["centroids", "polygons"], ...] = ("centroids", "polygons")
_IMAGE_ROLES: tuple[ImageRole, ...] = ("registered", "mask", "tissue_cut", "distance_mask", "expression")
# Images without a GEF or H5AD pitch keep their pixel grid, never `global`.
IMAGE_PIXELS = "image_pixels"


def _choice[T: str](value: T, name: str, choices: tuple[T, ...]) -> T:
    if not isinstance(value, str):
        message = f"{name} must be one of {list(choices)}; found {type(value).__name__}."
        raise TypeError(message)
    if value not in choices:
        message = f"{name} must be one of {list(choices)}; found {value!r}."
        raise ValueError(message)
    return value


def _sequence[T](values: Sequence[T], name: str) -> tuple[T, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        message = f"{name} must be a sequence, not a scalar or string."
        raise TypeError(message)
    return tuple(values)


def _sample_id(value: str | None) -> str | None:
    """Validate a sample selector; it selects files and never names elements."""
    if value is None:
        return None
    if not isinstance(value, str):
        message = f"sample_id must be a string; found {type(value).__name__}."
        raise TypeError(message)
    if not value or value in {".", ".."} or any(character in value for character in "/\\\0"):
        message = "sample_id must be a nonempty sample filename component."
        raise ValueError(message)
    return value


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


def _path(value: str | Path, name: str) -> str | Path:
    if not isinstance(value, (str, Path)):
        message = f"{name} must contain filesystem paths; found {type(value).__name__}."
        raise TypeError(message)
    return value


def _image_roles(values: Sequence[ImageRole]) -> tuple[ImageRole, ...]:
    roles = tuple(_choice(role, "image_roles", _IMAGE_ROLES) for role in _sequence(values, "image_roles"))
    if len(set(roles)) != len(roles):
        message = f"image_roles must be distinct; found {list(roles)}."
        raise ValueError(message)
    return roles


def _transformations(pitch_nm: int | None) -> dict[str, BaseTransformation]:
    """Map the shared DNB grid to micrometres, or keep it as a pixel frame without a pitch."""
    if pitch_nm is None:
        return {IMAGE_PIXELS: Identity()}
    return {"global": Scale([pitch_nm / 1000, pitch_nm / 1000], axes=("x", "y"))}


def _require_products(
    bins: tuple[int, ...] | Literal["all"],
    cell_bins: Literal["none", "unadjusted", "adjusted", "both"],
    roles: tuple[ImageRole, ...],
    analyses: tuple[str | Path, ...],
    overrides: tuple[str | Path | None, ...],
) -> None:
    """Require at least one product and a selected product for every count override."""
    if not bins and cell_bins == "none" and not roles and not analyses:
        message = "Select products, e.g. stereoseq(path, bin_sizes=(50,)) or stereoseq(path, cell_bins='unadjusted')."
        raise ValueError(message)
    square, cells, adjusted = (override is not None for override in overrides)
    if (
        (square and not bins)
        or (cells and cell_bins not in ("unadjusted", "both"))
        or (adjusted and cell_bins not in ("adjusted", "both"))
    ):
        message = "Count-source overrides require selection of their corresponding product."
        raise ValueError(message)


def _population(source: CellSource) -> Literal["cells", "adjusted_cells"]:
    return "cells" if source.kind == "unadjusted" else "adjusted_cells"


def _square_products(
    square: SquareCounts, transformations: Mapping[str, BaseTransformation]
) -> tuple[dd.DataFrame, ad.AnnData]:
    """Build one square bin's points and linked table; the IDs are the row positions."""
    name = f"bin{square.bin_size}"
    index = pd.RangeIndex(len(square.coordinates))
    context = ReaderErrorContext("stereoseq", "square counts", square.format_version)
    finalize_positions(
        square.table,
        square.resolution,
        "source_bin_reference_position",
        calibration=f"GEF geneExp/{name}/expression@resolution = {square.resolution} nm per grid unit",
        context=context,
        path=square.file,
    )
    points = make_points(square.coordinates, index, transformations=transformations)
    return points, link_table(square.table, f"{name}_genes", index, context=context, path=square.file)


def _cell_products(
    source: CellSource,
    cells: CellCounts,
    target: str,
    *,
    policy: InvalidGeometryPolicy,
    transformations: Mapping[str, BaseTransformation],
) -> tuple[dd.DataFrame | gpd.GeoDataFrame, ad.AnnData]:
    """Build one cell population's centroids or polygons and its linked table."""
    context = ReaderErrorContext("stereoseq", f"{source.kind} cell counts", cells.format_version)
    index = pd.Index(cells.cell_ids)
    element: dd.DataFrame | gpd.GeoDataFrame
    if cells.borders is None:
        element = make_points(cells.coordinates, index, transformations=transformations)
    else:
        element = make_polygons(
            cells.coordinates, cells.borders, index, path=source.file, policy=policy, transformations=transformations
        )
        # Dropped contours remove their rows; analyses still match the full population.
        index = element.index
    table = select_cells(cells.table, index, context=context, path=source.file)
    finalize_positions(
        table,
        cells.resolution,
        "vendor_cell_center",
        calibration=f"GEF root@resolution = {cells.resolution} nm per grid unit",
        context=context,
        path=source.file,
    )
    return element, link_table(table, target, index, context=context, path=source.file)


def _analysis_products(
    source: AnalysisSource,
    analysis: AnalysisResult,
    *,
    from_gef: bool,
    transformations: Mapping[str, BaseTransformation],
    duplicate_sums: list[SparseDuplicateSummary],
) -> tuple[dd.DataFrame, ad.AnnData]:
    """Build one analysis product's own points and linked table, in H5AD row order."""
    context = ReaderErrorContext("stereoseq", "analysis table")
    population = analysis.population
    finalize_positions(
        analysis.table,
        analysis.resolution,
        "vendor_cell_center" if population in ("cells", "adjusted_cells") else "source_bin_reference_position",
        calibration=(
            f"Matching raw GEF {population} resolution = {analysis.resolution} nm per grid unit"
            if from_gef
            else f"H5AD uns/resolution = {analysis.resolution} nm per grid unit"
        ),
        context=context,
        path=source.file,
    )
    points = make_points(analysis.coordinates, analysis.instance_ids, transformations=transformations)
    table = link_table(
        analysis.table,
        f"{analysis.name}_points",
        analysis.instance_ids,
        context=context,
        path=source.file,
        duplicate_sums=duplicate_sums,
    )
    return points, table


def _format_version(sources: Mapping[ArtifactFile, str | None]) -> str | None:
    """Return the GEF schema version shared by every read GEF, or ``None``.

    A file without a ``version`` attribute leaves the version unknown. Known
    versions that disagree are reported, because no single one describes the
    result.
    """
    versions = set(sources.values())
    if len(versions - {None}) > 1:
        found = ", ".join(f"{file.name}: {version or 'no version'}" for file, version in sources.items())
        warnings.warn(
            f"stereoseq: the read GEFs have different schema versions ({found}); no format version is recorded.",
            UserWarning,
            stacklevel=3,
        )
        return None
    return versions.pop() if len(versions) == 1 else None


def _validate_frames(
    products: Sequence[SquareCounts | CellCounts], sample: str | None, path: Path, format_version: str | None
) -> None:
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
            context=ReaderErrorContext("stereoseq", "coordinate frame", format_version),
            path=path,
            found="inconsistent source sample, origin or physical pitch",
            expected="matching normalized grids; separately recentered sources require an established transform",
        )


def stereoseq(  # noqa: PLR0913 - every product selection is an explicit public option.
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
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read selected Stereo-seq (SAW) count, analysis and image products.

    Supported inputs are the compact numbered workflow, the SAW 7.0/7.1
    numbered layouts, the SAW 8.0/8.1/8.2 ``outs`` layouts (task root or
    ``outs`` itself) and flat directories of exported files. The Stereo-seq
    format guide shows the directory trees.

    Provenance records no SAW release: the directory family does not identify
    one and the read artifacts do not store one. It records the GEF schema
    version as ``gef-<n>`` (the root ``version`` attribute, for example
    ``gef-4``) when every read square or cell GEF reports the same version.
    It records none when a GEF has no ``version``, when the versions differ,
    or when no count GEF is read (analysis or image products alone, even
    when a GEF supplies the image pitch).

    Nothing is read by default: select at least one product (square bins, cell
    bins, analysis files or image roles). Products are independent; for
    example, cells need no square counts, analysis or images.

    Parameters
    ----------
    path
        Dataset root, task root containing ``outs``, or ``outs`` itself.
    sample_id
        Chip serial that selects the source files when the dataset contains
        several samples; inferred when unique. Dots are kept. It does not name
        any element.
    bin_sizes
        Stored square bin sizes to read, or ``"all"``. Empty reads no square
        counts. Missing bins fail; none are derived from finer bins.
    cell_bins
        Read no cells, the unadjusted cells, the adjusted cells, or both.
    cell_geometry
        ``"centroids"`` returns the stored cell centres as points and never
        reads contours; ``"polygons"`` returns the stored contours as shapes.
    invalid_geometries
        For polygons: keep invalid contours, drop them together with their
        table rows, or raise. Malformed contour padding always fails. Analysis
        tables are matched against the full raw population and are not
        filtered.
    square_counts_file
        Square GEF to read instead of the default unique ``tissue`` or full
        GEF; whole-chip ``raw`` counts need this override.
    cell_counts_file, adjusted_cell_counts_file
        Cell GEFs to read instead of the discovered ones. Flat directories
        need these overrides. An override does not select its product.
    analysis_files
        Analysis H5ADs to read. Relative paths resolve against ``path``;
        absolute paths may point anywhere.
    image_roles
        Image products to read, every matching stain included:
        ``"registered"`` microscopy, ``"mask"``, ``"tissue_cut"``,
        ``"distance_mask"`` (expanded cell masks) and ``"expression"``
        (``bin1_img.tif``).
    scale_factors
        Integer factors of at least two, one per generated pyramid level;
        empty for a single scale. They apply to every image and mask.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes. Masks use
        the ``y, x`` sizes.
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
        The selected products, each with its own target element and table.

        Count products:

        - square bin ``N``: points ``bin<N>_genes`` and table ``bin<N>_table``.
          Rows are the distinct stored positions in order of first occurrence,
          with IDs ``0, 1, ...``; records with the same position and feature
          are summed. Features keep the GEF order, including features without
          counts.
        - cells: points ``cells_centroids`` or shapes ``cells_polygons`` and
          table ``cells_table``, in source cell order with the source integer
          cell IDs; ``adjusted_cells_*`` for adjusted cells.
        - ``X`` and the optional ``layers["exon"]`` are canonical CSR sparse
          arrays, float32 when every count is exact, otherwise float64;
          counts that neither represents exactly raise. Source summaries and
          ``uns["stereoseq"]`` (origin, pitch, cell-type names) are kept.

        Analysis products: points ``<population>_analysis[_r<resolution>]_points``
        and table ``..._table``, where ``<population>`` is ``bin<N>``, ``cells``
        or ``adjusted_cells`` and ``<resolution>`` the clustering resolution
        from the filename with ``.`` replaced by ``_``. The H5AD keeps its
        values, row and feature order, annotations, embeddings and ``raw``;
        ``X`` and the ``counts``, ``log1p`` and ``exon`` layers become CSR
        sparse arrays and duplicate sparse entries are summed. ``obs`` and
        ``var`` column names and top-level ``uns`` keys that SpatialData does
        not accept are sanitized; ``uns`` values and nested keys are kept.

        Images (``c, y, x``, source dtype and channel order) and mask labels
        (``y, x``, source values) are named ``<stain>_<role>`` (distance masks
        ``<stain>_distance_mask_<distance>``), or the role alone without a
        stain; ``bin1_img_tissue_cut.tif`` becomes ``expression_tissue_cut``.
        Channels are ``r, g, b(, a)`` only when the TIFF declares an RGB colour
        model, otherwise positional. Pixels stay lazy.

        Coordinate systems: points, shapes and rasters store native DNB-grid
        coordinates (registered images share that grid). ``global`` maps them
        to micrometres with ``Scale(pitch_nm / 1000)``, using the common pitch
        of the selected GEFs and H5ADs, or, for images alone, of the sample's
        GEFs in the dataset. Images without any pitch go to ``image_pixels``
        instead and have no ``global`` frame. Every table holds float64
        micrometres in ``obsm["spatial"]`` and the exact grid coordinates in
        ``obsm["spatial_grid"]``, described in
        ``uns["spatialdata_io_positions"]``. Chip origins are recorded, not
        applied. Tables link to their target through ``obs["region"]`` and
        ``obs["instance_id"]``.

    Warns
    -----
    UserWarning
        Once per analysis file whose sparse duplicate entries were summed.
        When an image name from a file name, an analysis ``obs``/``var``
        column name or a top-level analysis ``uns`` key is not valid in
        SpatialData; it is sanitized.
        Once when the read GEFs report different schema versions; no format
        version is recorded.

    Raises
    ------
    TypeError, ValueError
        If an option has the wrong type or value, no product is selected, or
        a count override is given without its product.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If a selected artifact is missing or has the wrong kind.
    ValueError
        A contextual ``ReaderFormatError`` for an ambiguous layout, a missing
        bin or image role, malformed GEF, H5AD or TIFF content, inconsistent
        samples, origins or pitches, analysis rows or features absent from the
        matching GEF, or invalid polygons under ``"raise"``. Malformed analysis
        expression raises ``ValueError`` from final linkage.
    """
    sample_id = _sample_id(sample_id)
    bins = _bin_selection(bin_sizes)
    cell_bins = _choice(cell_bins, "cell_bins", _CELL_BINS)
    cell_geometry = _choice(cell_geometry, "cell_geometry", _CELL_GEOMETRIES)
    policy = validate_invalid_geometry_policy(invalid_geometries)
    overrides = tuple(
        None if value is None else _path(value, name)
        for value, name in (
            (square_counts_file, "square_counts_file"),
            (cell_counts_file, "cell_counts_file"),
            (adjusted_cell_counts_file, "adjusted_cell_counts_file"),
        )
    )
    analyses = tuple(_path(file, "analysis_files") for file in _sequence(analysis_files, "analysis_files"))
    roles = _image_roles(image_roles)
    raster = make_raster_options(scale_factors, chunks)
    _require_products(bins, cell_bins, roles, analyses, overrides)

    layout = discover_layout(
        path,
        sample_id=sample_id,
        bins=bins,
        cell_bins=cell_bins,
        square_counts_file=overrides[0],
        cell_counts_file=overrides[1],
        adjusted_cell_counts_file=overrides[2],
        analysis_files=analyses,
        image_roles=roles,
        path_policy=path_policy,
    )
    square_counts = read_square(layout.square) if layout.square is not None else ()
    cell_counts = tuple(read_cells(source, read_borders=cell_geometry == "polygons") for source in layout.cells)
    products: tuple[SquareCounts | CellCounts, ...] = (*square_counts, *cell_counts)
    format_version = _format_version(
        {square.file: square.format_version for square in square_counts}
        | {source.file: cells.format_version for source, cells in zip(layout.cells, cell_counts, strict=True)}
    )
    _validate_frames(products, layout.sample_id, layout.root.path, format_version)

    # Raw populations are the independent targets that analyses are matched
    # against; no analysis row order determines a raw identity.
    populations: dict[str, SquareCounts | CellCounts] = {f"bin{square.bin_size}": square for square in square_counts}
    populations.update((_population(source), cells) for source, cells in zip(layout.cells, cell_counts, strict=True))
    analysis_results, pitch = read_analyses(
        layout.analyses, counts=populations, resolution=products[0].resolution if products else None
    )
    if pitch is None:
        pitch = read_image_pitch(layout.calibration, sample_id=layout.sample_id)
    transformations = _transformations(pitch)

    points: dict[str, dd.DataFrame] = {}
    shapes: dict[str, gpd.GeoDataFrame] = {}
    tables: dict[str, ad.AnnData] = {}

    for square in square_counts:
        name = f"bin{square.bin_size}"
        points[f"{name}_genes"], tables[f"{name}_table"] = _square_products(square, transformations)

    for source, cells in zip(layout.cells, cell_counts, strict=True):
        name = _population(source)
        element, tables[f"{name}_table"] = _cell_products(
            source, cells, f"{name}_{cell_geometry}", policy=policy, transformations=transformations
        )
        if isinstance(element, gpd.GeoDataFrame):
            shapes[f"{name}_{cell_geometry}"] = element
        else:
            points[f"{name}_{cell_geometry}"] = element

    duplicates: list[tuple[Path, list[SparseDuplicateSummary]]] = []
    for source, analysis in zip(layout.analyses, analysis_results, strict=True):
        sums: list[SparseDuplicateSummary] = []
        points[f"{analysis.name}_points"], tables[f"{analysis.name}_table"] = _analysis_products(
            source,
            analysis,
            from_gef=analysis.population in populations,
            transformations=transformations,
            duplicate_sums=sums,
        )
        duplicates.append((source.file, sums))

    result = SpatialData(
        images={
            image.name: read_image(image, raster=raster, transformations=transformations)
            for image in layout.images
            if image.kind == "image"
        },
        labels={
            image.name: read_mask(image, raster=raster, transformations=transformations)
            for image in layout.images
            if image.kind == "labels"
        },
        points=points,
        shapes=shapes,
        tables=tables,
    )
    set_reader_provenance(result.attrs, reader="stereoseq", format_version=format_version)
    for file, sums in duplicates:
        if sums:
            warnings.warn(f"stereoseq {file.name}: " + "; ".join(map(str, sums)), UserWarning, stacklevel=2)
    return result
