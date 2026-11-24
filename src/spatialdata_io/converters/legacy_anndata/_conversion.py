"""Export explicit position snapshots and import legacy spatial AnnData."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import PointsModel, TableModel
from spatialdata.transformations import Identity

from spatialdata_io._positions import POSITIONS_KEY, FixedFrame, PositionsMetadata
from spatialdata_io.converters.legacy_anndata._centroids import centroid_coordinates, linked_targets
from spatialdata_io.converters.legacy_anndata._images import read_images, select_library
from spatialdata_io.converters.legacy_anndata._schemas import (
    coordinate_array,
    normalize_positions,
    observed_frames,
    parse_positions,
    validate_in_memory,
)
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.table import TableLinkage, normalize_owned_anndata, parse_linked_table

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FrameSelection:
    """Identify an export frame, its units and any original-row selection."""

    frame: str
    units: Literal["micrometer", "pixel", "grid", "unknown"]
    rows: NDArray[np.int64] | None = None
    calibration: dict[str, str] | None = None


def _frame_selection(
    table: AnnData,
    metadata: PositionsMetadata,
    frame: str | None,
    *,
    units: Literal["micrometer", "pixel", "grid"] | None = None,
) -> FrameSelection | None:
    descriptions = metadata["coordinates"]
    primary = descriptions.get("spatial") or next(iter(descriptions.values()), None)
    if frame is None:
        if primary is None:
            return None
        frames = observed_frames(table, primary)
        if not frames and not table.n_obs:
            return None
        if len(frames) != 1:
            message = f"Select coordinate_system explicitly for observation frames {frames!r}."
            raise ValueError(message)
        frame = frames[0]
    matches = [description for description in descriptions.values() if frame in observed_frames(table, description)]
    if not matches:
        if units is None:
            message = (
                f"No table position declaration establishes coordinate_system {frame!r}. "
                "For centroid export in a frame defined on linked elements, pass coordinate_system_units."
            )
            raise ValueError(message)
        calibration = (
            {frame: "Caller declared coordinate_system_units='micrometer'."} if units == "micrometer" else None
        )
        return FrameSelection(frame, units, calibration=calibration)
    declared_units = {description["units"] for description in matches}
    if len(declared_units) != 1 or (units is not None and units not in declared_units):
        message = f"Conflicting position units are declared for coordinate_system {frame!r}."
        raise ValueError(message)
    description = primary if primary is not None and primary in matches else matches[0]
    rows = None
    if "frame_key" in description:
        observations = cast("pd.DataFrame", table.obs)
        rows = np.flatnonzero(observations[description["frame_key"]].eq(frame).to_numpy(dtype=bool))
    return FrameSelection(frame, description["units"], rows, description.get("calibration"))


def _table(sdata: SpatialData, name: str | None) -> AnnData:
    if name is None:
        if len(sdata.tables) != 1:
            message = "Specify table_name when SpatialData does not contain exactly one table."
            raise ValueError(message)
        name = next(iter(sdata.tables))
    if not isinstance(name, str) or not name or name not in sdata.tables:
        message = f"Unknown table_name {name!r}; available tables are {tuple(sdata.tables)!r}."
        raise ValueError(message)
    return sdata.tables[name]


def _derive_positions(
    sdata: SpatialData, table: AnnData, metadata: PositionsMetadata, selection: FrameSelection
) -> None:
    if selection.units == "unknown":
        message = "Centroid export requires an established physical, pixel or grid frame; units are unknown."
        raise ValueError(message)
    physical = selection.units == "micrometer"
    destination = "spatial" if physical else "spatial_centroids"
    preservation = "spatial_original" if physical and "spatial" in table.obsm else None
    if (preservation is not None and preservation in table.obsm) or (not physical and destination in table.obsm):
        message = f"Centroid export destination {preservation or destination!r} is occupied."
        raise ValueError(message)
    targets = linked_targets(sdata, table)
    coordinates = centroid_coordinates(targets, frame=selection.frame, n_obs=table.n_obs)
    descriptions = metadata["coordinates"]
    if preservation is not None:
        table.obsm[preservation] = table.obsm.pop("spatial")
        descriptions[preservation] = descriptions.pop("spatial")
    derived: FixedFrame = {
        "axes": ["x", "y"],
        "units": selection.units,
        "frame": selection.frame,
        "position": f"linked_element_centroid: {', '.join(target.name for target in targets)}",
    }
    if selection.calibration is not None:
        derived["calibration"] = selection.calibration
    table.obsm[destination] = coordinates
    descriptions[destination] = derived
    table.uns[POSITIONS_KEY] = metadata


def to_legacy_anndata(
    sdata: SpatialData,
    *,
    table_name: str | None = None,
    coordinate_system: str | None = None,
    position_source: Literal["table", "centroids"] = "table",
    coordinate_system_units: Literal["micrometer", "pixel", "grid"] | None = None,
) -> AnnData:
    """Export an owned AnnData with an explicit position source.

    Parameters
    ----------
    sdata
        SpatialData containing an eager in-memory table. Spatial elements remain
        borrowed and are never modified.
    table_name
        Table to export; inferred only when exactly one table exists.
    coordinate_system
        Destination frame. Inferred only for a single observed table frame.
        A frame declared by the table selects its rows in their original order.
        Centroid export to a frame defined only on the linked elements uses
        every row and requires ``coordinate_system_units``.
    position_source
        ``'table'`` preserves stored positions without computing spatial
        elements. ``'centroids'`` derives XY centres from linked Shapes, Labels
        or Points in the selected frame, matching region and instance IDs.
        Each referenced target needs a finite, invertible XY transformation
        to the selected frame. Labels can require a complete raster read and
        Points can require a full coordinate scan.
    coordinate_system_units
        Destination units for ``position_source='centroids'``. Required when
        the table has no declaration for the explicitly named frame. Units
        must agree with existing declarations for the frame. The output
        calibration records any caller-declared micrometre units for a new
        frame.

    Returns
    -------
    anndata.AnnData
        Independent table with canonical CSR sparse-array ``X``. Canonical
        ``spatial`` exists only for established finite float64 XY micrometres.
        Native ``spatial_pixels``, ``spatial_grid`` and ``spatial_source`` stay
        intact. Replaced physical positions and their declaration are preserved
        as ``spatial_original``; native derived centres use ``spatial_centroids``.
        Images are not exported. Other layers and ``raw`` retain source storage.

    Raises
    ------
    TypeError
        If options, annotations or array storage are unsupported or lazy.
    ValueError
        If table/frame selection, calibration, coordinates, identifiers or
        preservation destinations are malformed or ambiguous.

    Notes
    -----
    Positions are snapshots. Geometry centres differ from vendor cell centres;
    deriving them after registration does not refresh the vendor definition.
    Equal frame names do not register independent datasets. For ordinary
    analysis, use ``sdata.tables[table_name]`` directly and copy only to establish
    independent mutation ownership. Legacy image-library metadata is preserved
    as metadata and does not establish physical calibration.
    """
    if position_source not in ("table", "centroids"):
        message = "position_source must be 'table' or 'centroids'."
        raise ValueError(message)
    if coordinate_system is not None and (not isinstance(coordinate_system, str) or not coordinate_system.strip()):
        message = "coordinate_system must be a nonempty string or None."
        raise TypeError(message)
    if coordinate_system_units is not None:
        if not isinstance(coordinate_system_units, str) or coordinate_system_units not in (
            "micrometer",
            "pixel",
            "grid",
        ):
            message = "coordinate_system_units must be 'micrometer', 'pixel', 'grid' or None."
            raise ValueError(message)
        if position_source != "centroids":
            message = "coordinate_system_units is supported only with position_source='centroids'."
            raise ValueError(message)
    source = _table(sdata, table_name)
    validate_in_memory(source)
    metadata = parse_positions(source)
    selection = _frame_selection(source, metadata, coordinate_system, units=coordinate_system_units)
    selected = source if selection is None or selection.rows is None else source[selection.rows]
    table = selected.copy()
    normalize_positions(table, metadata)
    if position_source == "centroids":
        if selection is None:
            message = (
                "Centroid export requires an established coordinate frame or explicit coordinate_system and units."
            )
            raise ValueError(message)
        _derive_positions(sdata, table, metadata, selection)
    return normalize_owned_anndata(table)


def from_legacy_anndata(
    adata: AnnData,
    *,
    library_id: str | None = None,
    library_key: str | None = None,
    rgb: bool | None = None,
) -> SpatialData:
    """Import one legacy image library and its observation-aligned positions.

    Parameters
    ----------
    adata
        Borrowed eager in-memory AnnData. Plain legacy ``spatial`` means
        full-resolution pixel-index XY; it moves exactly to ``spatial_pixels``.
        A valid single-frame physical declaration is retained without images.
    library_id
        Selected legacy library; inferred only when unambiguous.
    library_key
        Observation column associating rows with libraries. Required when
        choosing among multiple libraries; always validated when supplied.
    rgb
        Interpret three/four-channel images as RGB(A) when ``None``. Pass
        ``False`` for arbitrary numerical channels or ``True`` for RGB(A).

    Returns
    -------
    spatialdata.SpatialData
        Owned ``table``, existing selected-library images at stored resolution,
        and a ``locations`` Points element when nonempty positions exist.
        New ``spatialdata_region``/``spatialdata_instance_id`` columns link
        original-row ordinal IDs without changing original identifiers.
        Canonical ``spatial`` is absent without physical calibration. ``X`` is
        a canonical CSR sparse array; other layers and ``raw`` are preserved.
        Empty/positionless tables remain unlinked. No footprints are invented.

    Raises
    ------
    TypeError
        If input storage, RGB option or consumed image metadata is unsupported.
    ValueError
        If coordinates, declarations, library associations, scale factors or
        generated destinations are malformed, conflicting or ambiguous.

    Notes
    -----
    Images are not resampled. For a legacy scale ``s``, their intrinsic raster
    centres map to legacy pixel indices as ``(intrinsic_xy - 0.5) / s``. Scale
    factors describe relative pixel resolution, not micrometre calibration.
    Physical coordinates with legacy images require an independently established
    mapping and are unsupported here. Caller buffers and metadata are unchanged.
    """
    if rgb is not None and not isinstance(rgb, bool):
        message = "rgb must be a boolean or None."
        raise TypeError(message)
    validate_in_memory(adata)
    metadata = parse_positions(adata)
    library = select_library(adata, library_id=library_id, library_key=library_key)
    has_positions = "spatial" in adata.obsm
    if has_positions:
        description = metadata["coordinates"].get("spatial")
        if description is None:
            coordinate_array(adata, "spatial")
            metadata["coordinates"]["spatial"] = {
                "axes": ["x", "y"],
                "units": "pixel",
                "frame": f"legacy_{library.name}_pixels" if library.name is not None else "legacy_pixels",
                "position": "legacy_full_resolution_pixel_index",
            }
        elif description["units"] not in ("pixel", "micrometer"):
            message = "Legacy spatial positions must declare pixel or micrometer units, not grid or unknown units."
            raise ValueError(message)
        if metadata["coordinates"]["spatial"]["units"] == "micrometer" and library.images:
            message = "Physical spatial positions with legacy images require an established physical image mapping."
            raise ValueError(message)
        if adata.n_obs and ("spatialdata_region" in adata.obs or "spatialdata_instance_id" in adata.obs):
            message = "Legacy import observation destinations spatialdata_region/spatialdata_instance_id are occupied."
            raise ValueError(message)
    selected = adata if library.rows is None else adata[library.rows]
    table = selected.copy()
    selection = _frame_selection(table, metadata, None) if has_positions else None
    normalize_positions(table, metadata)
    if selection is None and library.images and has_positions:
        message = "Empty row-frame positions do not establish an image frame; select an explicit input frame."
        raise ValueError(message)
    frame = selection.frame if selection is not None else f"legacy_{library.name}_pixels"
    images = read_images(table, library, frame=frame, rgb=rgb)
    table.uns.pop(TableModel.ATTRS_KEY, None)
    context = ReaderErrorContext("from_legacy_anndata", "table")
    normalize_owned_anndata(table, context=context)
    if not has_positions or not table.n_obs:
        return SpatialData(tables={"table": table}, images=images)
    key = "spatial" if "spatial" in table.obsm else "spatial_pixels"
    coordinates = cast("np.ndarray", table.obsm[key])
    ids = np.arange(table.n_obs, dtype=np.int64) if library.rows is None else library.rows
    locations = PointsModel.parse(
        pd.DataFrame({"x": coordinates[:, 0], "y": coordinates[:, 1]}, index=ids),
        transformations={frame: Identity()},
    )
    table.obs["spatialdata_region"] = pd.Categorical(["locations"] * table.n_obs)
    table.obs["spatialdata_instance_id"] = ids
    table = parse_linked_table(
        table,
        TableLinkage(("locations",), "spatialdata_region", "spatialdata_instance_id"),
        context=context,
        expected_instance_ids={"locations": pd.Index(ids)},
    )
    return SpatialData(tables={"table": table}, images=images, points={"locations": locations})
