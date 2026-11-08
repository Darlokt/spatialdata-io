"""Assemble one Xenium bundle into SpatialData with final exact table linkage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pandas as pd
from spatialdata import SpatialData
from xarray import DataArray, DataTree

from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import validate_invalid_geometry_policy
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.provenance import set_reader_provenance
from spatialdata_io.readers.xenium._aligned import read_aligned
from spatialdata_io.readers.xenium._analysis import read_analysis
from spatialdata_io.readers.xenium._cells import read_cells_store
from spatialdata_io.readers.xenium._geometry import (
    apply_policy,
    make_cell_polygons,
    make_circles,
    make_nucleus_polygons,
    read_polygons,
)
from spatialdata_io.readers.xenium._images import read_morphology
from spatialdata_io.readers.xenium._labels import read_labels
from spatialdata_io.readers.xenium._layout import Selection, discover_layout
from spatialdata_io.readers.xenium._metadata import read_gene_panel, read_metrics, read_protein_panel
from spatialdata_io.readers.xenium._schemas import integer_ids
from spatialdata_io.readers.xenium._tables import build_table, link_table, read_cells, read_counts, read_feature_index
from spatialdata_io.readers.xenium._transcripts import read_transcripts

if TYPE_CHECKING:
    from pathlib import Path

    import anndata as ad
    import dask.dataframe as dd
    import geopandas as gpd

    from spatialdata_io.readers._utils.geometry import InvalidGeometryPolicy
    from spatialdata_io.readers._utils.image import RasterOptions
    from spatialdata_io.readers.xenium._cells import CellStore
    from spatialdata_io.readers.xenium._layout import TableTarget, XeniumLayout
    from spatialdata_io.readers.xenium._metadata import ProteinPanel
    from spatialdata_io.readers.xenium._tables import Cells, Counts, FeatureSelection

type CellGeometry = Literal["polygons", "circles", "both"]

_TARGETS = ("cell_labels", "cell_boundaries", "cell_circles")
_GEOMETRIES = ("polygons", "circles", "both")


@dataclass(frozen=True, slots=True)
class _Options:
    """Validated public options of one read."""

    selection: Selection
    feature_types: FeatureSelection
    invalid_geometries: InvalidGeometryPolicy
    raster: RasterOptions


@dataclass(frozen=True, slots=True)
class _Shapes:
    """Returned shapes and the IDs of cells dropped with their invalid polygons."""

    elements: dict[str, gpd.GeoDataFrame]
    dropped: pd.Index


def _boolean(name: str, value: object) -> None:
    if not isinstance(value, bool):
        message = f"{name} must be a boolean; found {type(value).__name__}."
        raise TypeError(message)


def _choice(name: str, value: object, choices: tuple[str, ...]) -> None:
    if not isinstance(value, str):
        message = f"{name} must be one of {choices}; found {type(value).__name__}."
        raise TypeError(message)
    if value not in choices:
        message = f"{name} must be one of {choices}; found {value!r}."
        raise ValueError(message)


def _feature_types(value: object) -> FeatureSelection:
    if value is None or value == "all":
        return value
    if isinstance(value, str) or not isinstance(value, Sequence):
        message = "feature_types must be None, 'all' or a sequence of feature-type names."
        raise TypeError(message)
    selected = tuple(value)
    if not all(isinstance(name, str) for name in selected):
        message = "feature_types must contain feature-type names."
        raise TypeError(message)
    if not selected or not all(selected) or len(set(selected)) != len(selected):
        message = "feature_types must contain distinct nonempty feature-type names."
        raise ValueError(message)
    return selected


def _validate_options(  # noqa: PLR0913, RUF100 - Every public product and format option is checked here.
    *,
    include_table: bool,
    feature_types: Sequence[str] | Literal["all"] | None,
    table_target: TableTarget | None,
    include_cell_labels: bool,
    include_nucleus_labels: bool,
    include_cell_boundaries: bool,
    cell_geometry: CellGeometry,
    include_nucleus_boundaries: bool,
    invalid_geometries: InvalidGeometryPolicy,
    include_transcripts: bool,
    include_morphology_focus: bool,
    include_morphology_mip: bool,
    include_aligned_images: bool,
    include_analysis: bool,
    scale_factors: Sequence[int],
    chunks: int | tuple[int, int, int],
) -> _Options:
    """Check every option in signature order, then their combination, before any I/O."""
    _boolean("include_table", include_table)
    features = _feature_types(feature_types)
    if table_target is not None:
        _choice("table_target", table_target, _TARGETS)
    _boolean("include_cell_labels", include_cell_labels)
    _boolean("include_nucleus_labels", include_nucleus_labels)
    _boolean("include_cell_boundaries", include_cell_boundaries)
    _choice("cell_geometry", cell_geometry, _GEOMETRIES)
    _boolean("include_nucleus_boundaries", include_nucleus_boundaries)
    policy = validate_invalid_geometry_policy(invalid_geometries)
    _boolean("include_transcripts", include_transcripts)
    _boolean("include_morphology_focus", include_morphology_focus)
    _boolean("include_morphology_mip", include_morphology_mip)
    _boolean("include_aligned_images", include_aligned_images)
    _boolean("include_analysis", include_analysis)
    raster = make_raster_options(scale_factors, chunks)

    polygons = include_cell_boundaries and cell_geometry in ("polygons", "both")
    circles = include_cell_boundaries and cell_geometry in ("circles", "both")
    target = table_target if include_table else None
    returned = {"cell_labels": include_cell_labels, "cell_boundaries": polygons, "cell_circles": circles}
    if target is not None and not returned[target]:
        message = (
            f"table_target={target!r} links the table to an element that is not returned. Include it "
            "(include_cell_labels=True, or include_cell_boundaries=True with a matching cell_geometry), "
            "or choose another table_target or None."
        )
        raise ValueError(message)
    if include_analysis and not include_table:
        message = "include_analysis annotates the table and requires include_table=True."
        raise ValueError(message)
    products = (
        include_table,
        include_cell_labels,
        include_nucleus_labels,
        include_cell_boundaries,
        include_nucleus_boundaries,
        include_transcripts,
        include_morphology_focus,
        include_morphology_mip,
        include_aligned_images,
    )
    if not any(products):
        message = "Select at least one product, for example include_table=True."
        raise ValueError(message)

    selection = Selection(
        table=include_table,
        table_target=target,
        cell_labels=include_cell_labels,
        nucleus_labels=include_nucleus_labels,
        cell_polygons=polygons,
        cell_circles=circles,
        nucleus_polygons=include_nucleus_boundaries,
        transcripts=include_transcripts,
        morphology_focus=include_morphology_focus,
        morphology_mip=include_morphology_mip,
        aligned_images=include_aligned_images,
        analysis=include_analysis,
    )
    return _Options(selection, features, policy, raster)


def _shapes(
    layout: XeniumLayout, store: CellStore | None, cells: Cells | None, *, circles: bool, policy: InvalidGeometryPolicy
) -> _Shapes:
    """Apply the invalid-geometry policy once; an invalid cell polygon drops its whole entity."""
    family, pixel_size = layout.experiment.family, layout.experiment.pixel_size
    elements: dict[str, gpd.GeoDataFrame] = {}
    dropped = pd.Index([])

    if layout.cell_boundaries is not None:
        context = layout.experiment.context("cell boundaries")
        polygons = read_polygons(layout.cell_boundaries, family, "cell", store, context=context)
        kept = apply_policy(polygons, policy)
        dropped = polygons.cell_ids[~kept]
        elements["cell_boundaries"] = make_cell_polygons(polygons, kept, pixel_size)
    if layout.nucleus_boundaries is not None:
        context = layout.experiment.context("nucleus boundaries")
        nuclei = read_polygons(layout.nucleus_boundaries, family, "nucleus", store, context=context)
        kept = apply_policy(nuclei, policy) & ~nuclei.cell_ids.isin(dropped)
        elements["nucleus_boundaries"] = make_nucleus_polygons(nuclei, kept, pixel_size)
    if circles and cells is not None:
        elements["cell_circles"] = make_circles(cells, ~cells.ids.isin(dropped))
    return _Shapes(elements, dropped)


def _labels(
    layout: XeniumLayout, store: CellStore | None, selection: Selection, raster: RasterOptions
) -> dict[str, DataArray | DataTree]:
    labels: dict[str, DataArray | DataTree] = {}
    for role, requested in (("cell", selection.cell_labels), ("nucleus", selection.nucleus_labels)):
        if not requested:
            continue
        mask = None if store is None else store.masks.get(role)
        if mask is None:
            raise ReaderFormatError(
                context=layout.experiment.context(f"{role} labels"),
                path=layout.cells_store,
                found=f"no usable {role} mask (absent, or a single-polygon-set Xenium Ranger import)",
                expected=f"a {role} mask in cells.zarr.zip; use include_{role}_labels=False",
            )
        pixel_size = layout.experiment.pixel_size
        labels[f"{role}_labels"] = read_labels(mask, pixel_size=pixel_size, raster=raster, element=f"{role} labels")
    return labels


def _images(layout: XeniumLayout, raster: RasterOptions) -> dict[str, DataArray | DataTree]:
    pixel_size = layout.experiment.pixel_size
    images: dict[str, DataArray | DataTree] = {}
    for name, image, artifact in (
        ("morphology_focus", layout.focus, "morphology focus image"),
        ("morphology_mip", layout.mip, "maximum-intensity projection"),
    ):
        if image is not None:
            context = layout.experiment.context(artifact)
            images[name] = read_morphology(image, pixel_size=pixel_size, raster=raster, context=context)
    for aligned in layout.aligned:
        context = layout.experiment.context(f"aligned {aligned.kind.upper()} image")
        images[aligned.name] = read_aligned(
            aligned.file, aligned.alignment, pixel_size=pixel_size, data_axes=None, raster=raster, context=context
        )
    return images


def _scale0_shape(element: DataArray | DataTree) -> tuple[int, ...]:
    if isinstance(element, DataTree):
        return tuple(next(iter(element["scale0"].data_vars.values())).shape)
    return tuple(element.shape)


def _check_grid(rasters: dict[str, DataArray | DataTree], root: Path, context: ReaderErrorContext) -> None:
    """Masks and morphology images share the calibrated pixel grid."""
    grids = {name: _scale0_shape(element)[-2:] for name, element in rasters.items()}
    if len(set(grids.values())) > 1:
        raise ReaderFormatError(
            context=context,
            path=root,
            found=f"grids {grids}",
            expected="masks and morphology images on one pixel grid",
        )


def _transcripts(layout: XeniumLayout, counts: Counts | None) -> dd.DataFrame:
    """Use the count matrix vocabulary and barcodes; without a table, read only their names."""
    source, family = layout.transcripts, layout.experiment.family
    if source is None or layout.counts is None:  # pragma: no cover - discovery resolves both together
        message = "transcripts require a count source"
        raise ValueError(message)
    if counts is not None:
        features, barcodes = counts.features, counts.table.obs_names
    else:
        index = read_feature_index(layout.counts, context=layout.experiment.context("cell-feature matrix"))
        features, barcodes = index.features, index.barcodes
    if family == "integer_ids":
        context = layout.experiment.context("cell-feature matrix barcodes")
        barcodes = integer_ids(barcodes, path=source.file, context=context)
    context = layout.experiment.context("transcripts")
    return read_transcripts(source, family, features=features, cells=barcodes, context=context)


def _table(  # noqa: PLR0913, RUF100 - The table combines every table-side product.
    layout: XeniumLayout,
    counts: Counts,
    cells: Cells,
    *,
    store: CellStore | None,
    shapes: _Shapes,
    target: TableTarget | None,
) -> ad.AnnData:
    """Build the table, remove dropped entities and link it to its target as the final mutation."""
    analysis = None
    if layout.analysis is not None:
        analysis = read_analysis(layout.analysis, counts.table.obs_names, context=layout.experiment.context("analysis"))
    table = build_table(
        cells, counts, store, analysis, path=layout.root.path, context=layout.experiment.context("cells table")
    )

    keep = ~cells.ids.isin(shapes.dropped)
    if not keep.all():
        table = table[keep].copy()
    expected: pd.Index | None = None
    if target == "cell_labels" and store is not None:
        expected = pd.Index(store.cell_labels[keep])
    elif target is not None:
        expected = shapes.elements[target].index
    return link_table(table, target, expected, context=layout.experiment.context("table"))


def _attrs(layout: XeniumLayout, panel: ProteinPanel | None, counts: Counts | None) -> dict[str, object]:
    """Collect JSON-serializable dataset metadata; local paths are never stored."""
    attrs: dict[str, object] = {"experiment": dict(layout.experiment.attrs)}
    if layout.metrics is not None:
        attrs["metrics_summary"] = read_metrics(
            layout.metrics, context=layout.experiment.context("metrics_summary.csv")
        )
    if layout.gene_panel is not None:
        attrs["gene_panel"] = read_gene_panel(layout.gene_panel, context=layout.experiment.context("gene_panel.json"))
    if panel is not None:
        attrs["protein_panel"] = dict(panel.identity)
    if counts is not None and counts.protein is not None:
        attrs["protein"] = counts.protein
    return attrs


def xenium(  # noqa: PLR0913, RUF100 - Explicit supported products and format options form the public contract.
    path: str | Path,
    *,
    include_table: bool = True,
    feature_types: Sequence[str] | Literal["all"] | None = None,
    table_target: Literal["cell_labels", "cell_boundaries", "cell_circles"] | None = "cell_labels",
    include_cell_labels: bool = True,
    include_nucleus_labels: bool = True,
    include_cell_boundaries: bool = True,
    cell_geometry: Literal["polygons", "circles", "both"] = "polygons",
    include_nucleus_boundaries: bool = True,
    invalid_geometries: Literal["keep", "drop", "raise"] = "keep",
    include_transcripts: bool = True,
    include_morphology_focus: bool = True,
    include_morphology_mip: bool = False,
    include_aligned_images: bool = True,
    include_analysis: bool = False,
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    chunks: int | tuple[int, int, int] = DEFAULT_CHUNKS,
    path_policy: Literal["strict", "relaxed"] = "strict",
) -> SpatialData:
    """Read a 10x Genomics Xenium output bundle.

    Supports Xenium Onboard Analysis (XOA) 1.0 to 4.x bundles and Xenium Ranger
    outputs, in three families selected by the ``experiment.xenium`` format
    version: integer cell IDs (XOA 1.0-1.2), hexadecimal cell IDs with
    ``seg_mask_value`` (1.3-1.9) and polygon sets (2.0 onwards). Stripped
    deposits with CSV twins and MatrixMarket counts are read too. The Xenium
    format guide has the directory trees, migration notes and examples.

    Parameters
    ----------
    path
        Xenium output directory containing ``experiment.xenium``.
    include_table
        Return the cell table.
    feature_types
        Feature types in ``X``. ``None`` selects Gene Expression and, when
        present, Protein Expression; ``"all"`` keeps every type; a sequence
        selects exactly those types, each of which must exist. Order follows
        the source matrix.
    table_target
        Element the table annotates: ``"cell_labels"`` (instance column
        ``label_id``), ``"cell_boundaries"`` or ``"cell_circles"`` (instance
        column ``cell_id``); ``None`` returns an unlinked table. The target must
        be returned. Ignored without a table.
    include_cell_labels, include_nucleus_labels
        Return the segmentation masks of ``cells.zarr.zip``, the exact 10x
        segmentation. Labels are never derived from polygons.
    include_cell_boundaries
        Return cells as shapes in the representation chosen by ``cell_geometry``.
    cell_geometry
        ``"polygons"`` (``cell_boundaries``), ``"circles"`` (``cell_circles``,
        centroids with radius ``sqrt(cell_area / pi)``, so each circle has the
        exact mask area) or ``"both"``.
    include_nucleus_boundaries
        Return ``nucleus_boundaries``, indexed by nucleus mask label.
    invalid_geometries
        Boundary polygons are simplified by 10x and some are not valid
        geometries. The policy applies to the returned polygons. ``"keep"``
        returns them unchanged; ``"drop"`` removes a cell whose polygon is
        invalid together with its circle, nucleus polygons and table row, and an
        invalid nucleus polygon alone; ``"raise"`` rejects them. Without
        ``cell_boundaries`` no cell is dropped. Rings with fewer than four
        coordinates (Xenium Ranger imports) cannot be represented and require
        ``"drop"``. Labels and transcripts are never modified.
    include_transcripts
        Return lazy transcript points with every decoded transcript.
    include_morphology_focus
        Return ``morphology_focus``, the 2D focus image (all channels).
    include_morphology_mip
        Return ``morphology_mip``, the maximum-intensity projection that only
        XOA 1.x writes.
    include_aligned_images
        Return 10x-named post-Xenium images (``*he_image.ome.tif``,
        ``*if_image.ome.tif``) at the root, with their optional
        ``*imagealignment.csv``. Other images can be read with
        :func:`xenium_aligned_image`.
    include_analysis
        Add 10x secondary analysis to the table: clusterings as categorical
        ``obs`` columns named by method, PCA/UMAP projections as ``obsm``.
    scale_factors
        Integer factors of at least two, one per generated pyramid level of the
        images and labels; empty for a single scale.
    chunks
        One positive size for every axis or three ``c, y, x`` sizes of the
        images; labels use the ``y`` and ``x`` sizes. By default one channel per
        chunk and square 1024-pixel tiles.
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
        The requested elements, all in the ``global`` coordinate system in
        micrometres with its origin at the top-left corner of the morphology
        image. ``pixel_size`` is the ``experiment.xenium`` value.

        - Images ``morphology_focus`` and ``morphology_mip``: lazy ``c, y, x``
          in the source dtype with OME channel names, ``Scale(pixel_size)``.
        - Images ``he_image`` and ``if_image``: as :func:`xenium_aligned_image`.
        - Labels ``cell_labels`` and ``nucleus_labels``: lazy uint32 masks,
          ``Scale(pixel_size)``.
        - Shapes ``cell_boundaries`` (index ``cell_id``) and
          ``nucleus_boundaries`` (index ``label_id``, column ``cell_id``):
          source vertices traced through pixel centres, registered by a
          half-pixel ``Translation``.
        - Shapes ``cell_circles``: index ``cell_id``, column ``radius``,
          ``Identity()``.
        - Points ``transcripts``: lazy, ``Identity()``, with ``x, y, z``
          (float32 µm), categorical ``feature_name`` (count-matrix features in
          source order), ``cell_id`` with the vendor unassigned sentinel,
          ``qv`` and the other documented columns.
        - Table ``table``: see below.

        The table has an eager canonical CSR ``X`` in source feature order,
        float32 when every value is exact in it, otherwise float64; values
        neither type preserves raise. ``obs_names`` are the matrix barcodes.

        - ``obs``: the cells-table columns in source order, plus ``label_id``
          (uint32) and the nucleus centroids when ``cells.zarr.zip`` is
          present. Cells-table column names that SpatialData does not accept
          are renamed with a warning.
        - ``obsm["spatial"]``: float64 cell centroids in µm, described in
          ``uns["spatialdata_io_positions"]``.
        - ``var``: every selected source feature annotation, such as
          ``gene_ids``, ``feature_types`` and ``genome``. Feature names must
          be unique; unusable annotation names are renamed with a warning.
          Selected protein features add ``long_name``, ``clone``
          and ``codeword_index`` from ``protein_panel.json``; source columns
          with those names raise instead of being overwritten.

        ``attrs["xenium"]`` holds the ``experiment``, ``metrics_summary``,
        ``gene_panel``, ``protein_panel`` and ``protein`` metadata, and the
        provenance records the ``experiment.xenium`` format version. Pixels
        and transcripts stay lazy; the table and shapes are eager.

    Raises
    ------
    TypeError, ValueError
        If options are invalid or inconsistent.
    FileNotFoundError, NotADirectoryError, IsADirectoryError
        If a required artifact is missing or of the wrong kind.
    ValueError
        A contextual ``ReaderFormatError`` if the bundle violates its documented
        contract: unsupported versions, ambiguous twins, gzip-only transcripts,
        identity or label mismatches, malformed tables, geometry or images, or
        protein without the HDF5 scaling factor. Its context names the detected
        format version.

    Warns
    -----
    UserWarning
        If cells-table columns, HDF5 feature annotations or analysis
        clustering methods are renamed to usable SpatialData column names.

    Notes
    -----
    Protein Expression values are the mean stain intensity per cell after the
    100-count offset was removed, divided once by the float32 value of the HDF5
    ``protein_scaling_factor``. The count matrix
    contains assigned transcripts with ``qv >= 20``; filter points with
    ``points[points.qv >= 20]`` to match.
    """
    options = _validate_options(
        include_table=include_table,
        feature_types=feature_types,
        table_target=table_target,
        include_cell_labels=include_cell_labels,
        include_nucleus_labels=include_nucleus_labels,
        include_cell_boundaries=include_cell_boundaries,
        cell_geometry=cell_geometry,
        include_nucleus_boundaries=include_nucleus_boundaries,
        invalid_geometries=invalid_geometries,
        include_transcripts=include_transcripts,
        include_morphology_focus=include_morphology_focus,
        include_morphology_mip=include_morphology_mip,
        include_aligned_images=include_aligned_images,
        include_analysis=include_analysis,
        scale_factors=scale_factors,
        chunks=chunks,
    )
    selection = options.selection
    layout = discover_layout(path, selection, path_policy=path_policy)
    experiment = layout.experiment

    store = None
    if layout.cells_store is not None:
        context = experiment.context("cells.zarr.zip")
        store = read_cells_store(
            layout.cells_store, experiment.family, pixel_size=experiment.pixel_size, context=context
        )
    cells = (
        None
        if layout.cells is None
        else read_cells(layout.cells, experiment.family, context=experiment.context("cells table"))
    )
    panel = None
    if layout.protein_panel is not None:
        panel = read_protein_panel(layout.protein_panel, context=experiment.context("protein_panel.json"))
    counts = None
    if layout.counts is not None and selection.table:
        context = experiment.context("cell-feature matrix")
        counts = read_counts(layout.counts, options.feature_types, panel, context=context)

    shapes = _shapes(layout, store, cells, circles=selection.cell_circles, policy=options.invalid_geometries)
    labels = _labels(layout, store, selection, options.raster)
    images = _images(layout, options.raster)
    morphology = {name: image for name, image in images.items() if name.startswith("morphology")}
    _check_grid(labels | morphology, layout.root.path, experiment.context("pixel grid"))
    points = {} if layout.transcripts is None else {"transcripts": _transcripts(layout, counts)}
    tables = {}
    if counts is not None and cells is not None:
        tables["table"] = _table(layout, counts, cells, store=store, shapes=shapes, target=selection.table_target)

    result = SpatialData(images=images, labels=labels, points=points, shapes=shapes.elements, tables=tables)
    result.attrs["xenium"] = _attrs(layout, panel, counts)
    set_reader_provenance(result.attrs, reader="xenium", format_version=experiment.version)
    return result
