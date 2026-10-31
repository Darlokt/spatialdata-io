"""Assemble one Xenium bundle into SpatialData with final exact table linkage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pandas as pd
from spatialdata import SpatialData
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.geometry import validate_invalid_geometry_policy
from spatialdata_io.readers._utils.image import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS, make_raster_options
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
from spatialdata_io.readers.xenium._layout import FocusImages, Selection, discover_layout
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
class _Shapes:
    """Returned shapes and the IDs of cells dropped with their invalid polygons."""

    elements: dict[str, gpd.GeoDataFrame]
    dropped: pd.Index


def _feature_types(value: Sequence[str] | Literal["all"] | None) -> FeatureSelection:
    if value is None:
        return None
    if isinstance(value, str) and value == "all":
        return "all"
    if isinstance(value, str) or not isinstance(value, Sequence):
        message = "feature_types must be None, 'all' or a sequence of feature-type names."
        raise TypeError(message)
    selected = tuple(value)
    if (
        not selected
        or any(not isinstance(name, str) or not name for name in selected)
        or len(set(selected)) != len(selected)
    ):
        message = "feature_types must contain distinct nonempty feature-type names."
        raise ValueError(message)
    return selected


def _selection(  # noqa: PLR0913, RUF100 - Normalize the explicit public product choices together.
    *,
    include_table: bool,
    table_target: TableTarget | None,
    include_cell_labels: bool,
    include_nucleus_labels: bool,
    include_cell_boundaries: bool,
    cell_geometry: CellGeometry,
    include_nucleus_boundaries: bool,
    include_transcripts: bool,
    include_morphology_focus: bool,
    include_morphology_mip: bool,
    include_aligned_images: bool,
    include_analysis: bool,
) -> Selection:
    """Validate inclusion choices and one unambiguous table target before any I/O."""
    flags = {
        "include_table": include_table,
        "include_cell_labels": include_cell_labels,
        "include_nucleus_labels": include_nucleus_labels,
        "include_cell_boundaries": include_cell_boundaries,
        "include_nucleus_boundaries": include_nucleus_boundaries,
        "include_transcripts": include_transcripts,
        "include_morphology_focus": include_morphology_focus,
        "include_morphology_mip": include_morphology_mip,
        "include_aligned_images": include_aligned_images,
        "include_analysis": include_analysis,
    }
    if any(not isinstance(value, bool) for value in flags.values()):
        message = f"{next(name for name, value in flags.items() if not isinstance(value, bool))} must be a boolean."
        raise TypeError(message)
    if cell_geometry not in _GEOMETRIES:
        message = f"cell_geometry must be one of {_GEOMETRIES}."
        raise ValueError(message)
    if table_target is not None and table_target not in _TARGETS:
        message = f"table_target must be one of {_TARGETS} or None."
        raise ValueError(message)

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
    if not any(flags.values()):
        message = "Select at least one product, for example include_table=True."
        raise ValueError(message)

    return Selection(
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


def _shapes(
    layout: XeniumLayout,
    store: CellStore | None,
    cells: Cells | None,
    *,
    circles: bool,
    policy: InvalidGeometryPolicy,
) -> _Shapes:
    """Apply the invalid-geometry policy once; an invalid cell polygon drops its whole entity."""
    family, pixel_size = layout.experiment.family, layout.experiment.pixel_size
    elements: dict[str, gpd.GeoDataFrame] = {}
    dropped = pd.Index([])

    if layout.cell_boundaries is not None:
        polygons = read_polygons(layout.cell_boundaries, family, "cell", store)
        kept = apply_policy(polygons, policy, "cell")
        dropped = polygons.cell_ids[~kept]
        elements["cell_boundaries"] = make_cell_polygons(polygons, kept, pixel_size)
    if layout.nucleus_boundaries is not None:
        nuclei = read_polygons(layout.nucleus_boundaries, family, "nucleus", store)
        kept = apply_policy(nuclei, policy, "nucleus") & ~nuclei.cell_ids.isin(dropped)
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
                context=ReaderErrorContext("xenium", f"{role} labels"),
                path=layout.cells_store,
                found=f"no usable {role} mask (absent, or a single-polygon-set Xenium Ranger import)",
                expected=f"a {role} mask in cells.zarr.zip; use include_{role}_labels=False",
            )
        labels[f"{role}_labels"] = read_labels(mask, pixel_size=layout.experiment.pixel_size, raster=raster)
    return labels


def _images(layout: XeniumLayout, raster: RasterOptions) -> dict[str, DataArray | DataTree]:
    pixel_size = layout.experiment.pixel_size
    images: dict[str, DataArray | DataTree] = {}
    if layout.focus is not None:
        images["morphology_focus"] = read_morphology(layout.focus, pixel_size=pixel_size, raster=raster)
    if layout.mip is not None:
        mip = FocusImages(layout.mip, (layout.mip,))
        images["morphology_mip"] = read_morphology(mip, pixel_size=pixel_size, raster=raster)
    for aligned in layout.aligned:
        images[aligned.name] = read_aligned(
            aligned.file, aligned.alignment, pixel_size=pixel_size, data_axes=None, raster=raster
        )
    return images


def _scale0_shape(element: DataArray | DataTree) -> tuple[int, ...]:
    if isinstance(element, DataTree):
        return tuple(next(iter(element["scale0"].data_vars.values())).shape)
    return tuple(element.shape)


def _check_grid(rasters: dict[str, DataArray | DataTree], root: Path) -> None:
    """Masks and morphology images share the calibrated pixel grid."""
    grids = {name: _scale0_shape(element)[-2:] for name, element in rasters.items()}
    if len(set(grids.values())) > 1:
        raise ReaderFormatError(
            context=ReaderErrorContext("xenium", "pixel grid"),
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
        index = read_feature_index(layout.counts)
        features, barcodes = index.features, index.barcodes
    if family == "integer_ids":
        barcodes = integer_ids(barcodes, path=source.file, artifact="cell-feature matrix barcodes")
    return read_transcripts(source, family, features=features, cells=barcodes)


def _table(layout: XeniumLayout, counts: Counts, cells: Cells, store: CellStore | None) -> ad.AnnData:
    """Combine counts and annotations, then add the optional secondary analysis."""
    table = build_table(cells, counts, store, path=layout.root)
    if layout.analysis is None:
        return table
    analysis = read_analysis(layout.analysis, table.obs_names)
    conflicts = [method for method in analysis.clusters if method in table.obs]
    if conflicts:
        raise ReaderFormatError(
            context=ReaderErrorContext("xenium", "analysis"),
            path=layout.root,
            found=f"clusterings {conflicts!r} named like cell annotations",
            expected="distinct analysis and annotation names",
        )
    for method, values in analysis.clusters.items():
        table.obs[method] = values
    for name, values in analysis.projections.items():
        table.obsm[name] = values
    return table


def _link(
    table: ad.AnnData, cells: Cells, target: TableTarget | None, store: CellStore | None, shapes: _Shapes
) -> ad.AnnData:
    """Remove dropped entities and link the table to its target as the final mutation."""
    keep = ~cells.ids.isin(shapes.dropped)
    if not keep.all():
        table = table[keep].copy()

    expected: pd.Index | None = None
    if target == "cell_labels" and store is not None:
        expected = pd.Index(store.cell_labels[keep])
    elif target is not None:
        expected = shapes.elements[target].index
    return link_table(table, target, expected)


def _attrs(layout: XeniumLayout, panel: ProteinPanel | None, counts: Counts | None) -> dict[str, object]:
    """Collect JSON-serializable dataset metadata; local paths are never stored."""
    attrs: dict[str, object] = {"experiment": dict(layout.experiment.attrs)}
    if layout.metrics is not None:
        attrs["metrics_summary"] = read_metrics(layout.metrics)
    if layout.gene_panel is not None:
        attrs["gene_panel"] = read_gene_panel(layout.gene_panel)
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
) -> SpatialData:
    """Read a 10x Genomics Xenium output bundle.

    Supports Xenium Onboard Analysis (XOA) 1.0 to 4.x bundles and Xenium Ranger
    outputs, in three families selected by the ``experiment.xenium`` format
    version: integer cell IDs (XOA 1.0-1.2), hexadecimal cell IDs with
    ``seg_mask_value`` (1.3-1.9) and polygon sets (2.0 onwards). Stripped
    deposits with CSV twins and MatrixMarket counts are read too. See the
    Xenium format guide for directory trees, migration notes and examples.

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

    Returns
    -------
    spatialdata.SpatialData
        Requested elements, all in the ``global`` coordinate system in
        micrometres with its origin at the top-left image corner:

        - images ``morphology_focus``/``morphology_mip`` (``c, y, x``, OME
          channel names, source dtype) and ``he_image``/``if_image``;
        - labels ``cell_labels``/``nucleus_labels`` (uint32);
        - shapes ``cell_boundaries`` (index ``cell_id``), ``cell_circles``
          (index ``cell_id``, column ``radius``) and ``nucleus_boundaries``
          (index ``label_id``, column ``cell_id``);
        - points ``transcripts`` with ``x, y, z`` (float32 µm), categorical
          ``feature_name`` (count-matrix features in source order),
          ``cell_id`` with the vendor unassigned sentinel, ``qv`` and the other
          documented columns;
        - table ``table``: canonical CSR ``X`` in source feature order (float32
          for official int32 HDF5 and MTX inputs),
          ``obs`` with the cells-table columns, ``label_id`` (uint32) and nucleus
          centroids when ``cells.zarr.zip`` is present, and float64 µm centroids in
          ``obsm["spatial"]``, described as unchanged vendor cell-centroid XY
          micrometres in ``global`` by ``uns["spatialdata_io_positions"]``.
          Nullable nucleus summaries do not replace canonical cell centres.

        Alternate HDF5 numerical dtypes follow Scanpy; protein division requires
        floating storage. Float32 retains its integer precision limits.

        Images and labels carry ``Scale(pixel_size)``, polygons a half-pixel
        ``Translation`` registering their vertices (traced through pixel centres
        in index coordinates) and circles and transcripts identity.
        ``attrs["xenium"]`` holds ``experiment``, ``metrics_summary``,
        ``gene_panel``, ``protein_panel`` and ``protein`` metadata. Pixels and
        transcripts stay lazy; the table and shapes are eager.

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
        protein without the HDF5 scaling factor.

    Notes
    -----
    Protein Expression values are the mean stain intensity per cell after the
    100-count offset was removed, divided once in float32 by the HDF5
    ``protein_scaling_factor``. The count matrix contains assigned transcripts
    with ``qv >= 20``; filter points with ``points[points.qv >= 20]`` to match.
    """
    selection = _selection(
        include_table=include_table,
        table_target=table_target,
        include_cell_labels=include_cell_labels,
        include_nucleus_labels=include_nucleus_labels,
        include_cell_boundaries=include_cell_boundaries,
        cell_geometry=cell_geometry,
        include_nucleus_boundaries=include_nucleus_boundaries,
        include_transcripts=include_transcripts,
        include_morphology_focus=include_morphology_focus,
        include_morphology_mip=include_morphology_mip,
        include_aligned_images=include_aligned_images,
        include_analysis=include_analysis,
    )
    selected_features = _feature_types(feature_types)
    policy = validate_invalid_geometry_policy(invalid_geometries)
    raster = make_raster_options(scale_factors, chunks)

    layout = discover_layout(path, selection)
    experiment = layout.experiment
    version = None if experiment.version is None else str(experiment.version)

    store = (
        None
        if layout.cells_store is None
        else read_cells_store(
            layout.cells_store,
            experiment.family,
            pixel_size=experiment.pixel_size,
            context=ReaderErrorContext("xenium", "cells.zarr.zip", version),
        )
    )
    cells = None if layout.cells is None else read_cells(layout.cells, experiment.family)
    panel = None if layout.protein_panel is None else read_protein_panel(layout.protein_panel)
    counts = (
        None if layout.counts is None or not selection.table else read_counts(layout.counts, selected_features, panel)
    )

    shapes = _shapes(layout, store, cells, circles=selection.cell_circles, policy=policy)
    labels = _labels(layout, store, selection, raster)
    images = _images(layout, raster)
    _check_grid(labels | {name: image for name, image in images.items() if name.startswith("morphology")}, layout.root)
    points = {} if layout.transcripts is None else {"transcripts": _transcripts(layout, counts)}
    tables = {}
    if counts is not None and cells is not None:
        table = _table(layout, counts, cells, store)
        tables["table"] = _link(table, cells, selection.table_target, store, shapes)

    result = SpatialData(images=images, labels=labels, points=points, shapes=shapes.elements, tables=tables)
    result.attrs["xenium"] = _attrs(layout, panel, counts)
    set_reader_provenance(result.attrs, reader="xenium", format_version=version)
    return result
