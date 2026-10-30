"""Describe coordinate snapshots without a runtime coordinate framework."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

POSITIONS_KEY = "spatialdata_io_positions"


class CoordinateDescription(TypedDict):
    """Describe established XY semantics; source interpretation stays reader-owned."""

    axes: list[Literal["x", "y"]]
    units: Literal["micrometer", "pixel", "grid", "unknown"]
    position: str
    calibration: NotRequired[dict[str, str]]


class FixedFrame(CoordinateDescription):
    """Identify one frame without allocating a constant observation column."""

    frame: str


class RowFrames(CoordinateDescription):
    """Reference observation-aligned categorical frame identifiers."""

    frame_key: str


class PositionsMetadata(TypedDict):
    """Declare project-owned snapshot semantics, independently of table linkage."""

    schema_version: Literal[1]
    coordinates: dict[str, FixedFrame | RowFrames]
