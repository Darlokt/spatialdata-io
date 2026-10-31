"""Validate eager AnnData inputs and explicit coordinate descriptions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    from anndata import AnnData

from spatialdata_io._positions import POSITIONS_KEY, FixedFrame, PositionsMetadata, RowFrames

_XY_AXES = ("x", "y")


def validate_in_memory(table: AnnData) -> None:
    """Reject backed or lazy slots before taking an owned AnnData copy."""
    if table.isbacked:
        msg = "Legacy AnnData conversion requires an in-memory AnnData, not a backed object."
        raise ValueError(msg)
    if not isinstance(table.obs, pd.DataFrame) or not isinstance(table.var, pd.DataFrame):
        msg = "Legacy AnnData conversion requires eager pandas DataFrame observation and feature metadata."
        raise TypeError(msg)
    _eager_array(table.X, label="X", allow_none=True)
    for slot, values in (
        ("layers", table.layers),
        ("obsm", table.obsm),
        ("varm", table.varm),
        ("obsp", table.obsp),
        ("varp", table.varp),
    ):
        for key, value in values.items():
            _eager_array(value, label=f"{slot}[{key!r}]")
    if table.raw is not None:
        if not isinstance(table.raw.var, pd.DataFrame):
            msg = "Legacy AnnData conversion requires eager pandas DataFrame raw feature metadata."
            raise TypeError(msg)
        _eager_array(table.raw.X, label="raw.X", allow_none=True)
        # AnnData types the shared descriptor owner as AnnData although Raw
        # intentionally exposes the same eager aligned mapping.
        for key, value in table.raw.varm.items():  # pyrefly: ignore[bad-argument-type]
            _eager_array(value, label=f"raw.varm[{key!r}]")


def _eager_array(value: object, *, label: str, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, (np.ndarray, pd.DataFrame, sparse.sparray, sparse.spmatrix)):
        msg = f"Legacy AnnData conversion requires eager NumPy, pandas or SciPy data in {label}."
        raise TypeError(msg)


def coordinate_array(table: AnnData, key: str) -> npt.NDArray[np.number]:
    """Validate and borrow one declared finite numerical XY embedding."""
    if key not in table.obsm:
        msg = f"Coordinate declaration references missing obsm[{key!r}]."
        raise ValueError(msg)
    values = table.obsm[key]
    if not isinstance(values, np.ndarray) or values.dtype.kind not in "iuf":
        msg = f"obsm[{key!r}] must be an eager real numerical NumPy array."
        raise TypeError(msg)
    if values.shape != (table.n_obs, 2):
        msg = f"obsm[{key!r}] has shape {values.shape}; expected ({table.n_obs}, 2) XY coordinates."
        raise ValueError(msg)
    if not np.isfinite(values).all():
        msg = f"obsm[{key!r}] contains nonfinite coordinates."
        raise ValueError(msg)
    # The numerical dtype kind has been established at this external boundary.
    return cast("npt.NDArray[np.number]", values)


def physical_coordinates(coordinates: npt.NDArray[np.number], *, key: str) -> npt.NDArray[np.float64]:
    """Convert validated finite coordinates to float64 without losing values."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = coordinates.astype(np.float64, copy=False)
    if coordinates.dtype.kind in "iu":
        # NumPy float/integer comparisons can round both sides. Python integers
        # retain exactness for the uncommon values outside float64's safe range.
        outside = (
            (coordinates > 2**53) | (coordinates < -(2**53)) if coordinates.dtype.kind == "i" else coordinates > 2**53
        )
        exact = all(
            int(source) == int(target) for source, target in zip(coordinates[outside], converted[outside], strict=True)
        )
    else:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            exact = bool(np.equal(converted.astype(coordinates.dtype, copy=False), coordinates).all())
    if not exact:
        msg = f"obsm[{key!r}] cannot be represented losslessly as float64 micrometre coordinates."
        raise ValueError(msg)
    return converted


def parse_positions(table: AnnData) -> PositionsMetadata:
    """Validate external coordinate metadata and return an owned typed description."""
    if POSITIONS_KEY not in table.uns:
        return {"schema_version": 1, "coordinates": {}}
    metadata = _mapping(table.uns[POSITIONS_KEY], label=POSITIONS_KEY)
    _fields(metadata, required={"schema_version", "coordinates"}, optional=set(), label=POSITIONS_KEY)
    version = metadata["schema_version"]
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)) or version != 1:
        msg = f"{POSITIONS_KEY} requires schema_version 1; found {version!r}."
        raise ValueError(msg)
    coordinates = _mapping(metadata["coordinates"], label=f"{POSITIONS_KEY}.coordinates")
    result: PositionsMetadata = {"schema_version": 1, "coordinates": {}}
    for key, value in coordinates.items():
        if not key:
            msg = "Coordinate description keys must be nonempty strings."
            raise ValueError(msg)
        description = _description(table, value, key=key)
        coordinate_array(table, key)
        if description["units"] == "micrometer":
            frames = observed_frames(table, description)
            calibration = description.get("calibration", {})
            missing = set(frames).difference(calibration)
            if missing:
                msg = f"Physical obsm[{key!r}] requires calibration for frames {sorted(missing)!r}."
                raise ValueError(msg)
        result["coordinates"][key] = description
    return result


def _description(table: AnnData, value: object, *, key: str) -> FixedFrame | RowFrames:
    label = f"{POSITIONS_KEY}.coordinates[{key!r}]"
    description = _mapping(value, label=label)
    _fields(
        description,
        required={"axes", "units", "position"},
        optional={"frame", "frame_key", "calibration"},
        label=label,
    )
    axes = description["axes"]
    if isinstance(axes, np.ndarray) and axes.ndim == 1:
        axes = axes.tolist()
    if (
        not isinstance(axes, list)
        or len(axes) != len(_XY_AXES)
        or not all(isinstance(axis, str) and axis == expected for axis, expected in zip(axes, _XY_AXES, strict=True))
    ):
        msg = f"{label}.axes must be ['x', 'y'] in XY order."
        raise ValueError(msg)
    units = description["units"]
    if not isinstance(units, str) or units not in ("micrometer", "pixel", "grid", "unknown"):
        msg = f"{label}.units must be micrometer, pixel, grid or unknown; found {units!r}."
        raise ValueError(msg)
    position = _string(description["position"], label=f"{label}.position")
    if ("frame" in description) == ("frame_key" in description):
        msg = f"{label} must declare exactly one of frame or frame_key."
        raise ValueError(msg)
    result: FixedFrame | RowFrames
    if "frame" in description:
        result = {
            "axes": ["x", "y"],
            "units": cast("Literal['micrometer', 'pixel', 'grid', 'unknown']", units),
            "position": position,
            "frame": _string(description["frame"], label=f"{label}.frame"),
        }
    else:
        frame_key = _string(description["frame_key"], label=f"{label}.frame_key")
        if not isinstance(table.obs, pd.DataFrame):
            msg = f"{label}.frame_key requires eager pandas observation metadata."
            raise TypeError(msg)
        if frame_key not in table.obs:
            msg = f"{label}.frame_key references missing obs[{frame_key!r}]."
            raise ValueError(msg)
        frames = table.obs[frame_key]
        if (
            not isinstance(frames, pd.Series)
            or not isinstance(frames.dtype, pd.CategoricalDtype)
            or frames.isna().any()
            or any(not isinstance(frame, str) or not frame.strip() for frame in frames.cat.categories)
        ):
            msg = f"obs[{frame_key!r}] must contain nonnull categorical string frame identifiers."
            raise ValueError(msg)
        result = {
            "axes": ["x", "y"],
            "units": cast("Literal['micrometer', 'pixel', 'grid', 'unknown']", units),
            "position": position,
            "frame_key": frame_key,
        }
    if "calibration" in description:
        calibration = _mapping(description["calibration"], label=f"{label}.calibration")
        result["calibration"] = {
            _string(frame, label=f"{label}.calibration frame"): _string(reason, label=f"{label}.calibration[{frame!r}]")
            for frame, reason in calibration.items()
        }
    return result


def observed_frames(table: AnnData, description: FixedFrame | RowFrames) -> tuple[str, ...]:
    """Return declared fixed or observed row frames without allocating a constant column."""
    if "frame" in description:
        # TypedDict key membership does not narrow this structurally overlapping union.
        return (cast("FixedFrame", description)["frame"],)
    # parse_positions establishes eager pandas annotations and categorical strings.
    observations = cast("pd.DataFrame", table.obs)
    return tuple(observations[description["frame_key"]].unique().tolist())


def normalize_positions(table: AnnData, metadata: PositionsMetadata) -> PositionsMetadata:
    """Normalize parsed declarations and position destinations on an owned table."""
    descriptions = metadata["coordinates"]
    if "spatial" in table.obsm and ("spatial" not in descriptions or descriptions["spatial"]["units"] != "micrometer"):
        description = descriptions.get("spatial")
        units = description["units"] if description is not None else "unknown"
        destination = {"pixel": "spatial_pixels", "grid": "spatial_grid", "unknown": "spatial_source"}[units]
        if destination in table.obsm or destination in descriptions:
            msg = f"Cannot preserve obsm['spatial']: destination {destination!r} is occupied."
            raise ValueError(msg)
        table.obsm[destination] = table.obsm.pop("spatial")
        if description is not None:
            descriptions[destination] = descriptions.pop("spatial")
    for key, description in descriptions.items():
        if description["units"] == "micrometer":
            # parse_positions already validated the array shape, dtype and finiteness.
            coordinates = cast("npt.NDArray[np.number]", table.obsm[key])
            table.obsm[key] = physical_coordinates(coordinates, key=key)
    if descriptions:
        table.uns[POSITIONS_KEY] = metadata
    else:
        table.uns.pop(POSITIONS_KEY, None)
    return metadata


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        msg = f"{label} must be a mapping with string keys."
        raise TypeError(msg)
    # External mapping keys were validated above; values remain untrusted.
    return cast("Mapping[str, object]", value)


def _fields(values: Mapping[str, object], *, required: set[str], optional: set[str], label: str) -> None:
    missing = required.difference(values)
    extra = set(values).difference(required, optional)
    if missing or extra:
        msg = f"{label} has missing fields {sorted(missing)!r} and unsupported fields {sorted(extra)!r}."
        raise ValueError(msg)


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{label} must be a nonempty string."
        raise ValueError(msg)
    return value
