"""Deterministic CosMx flat-file exports with independent geometric oracles."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import imagecodecs
import numpy as np
import pandas as pd
import tifffile

if TYPE_CHECKING:
    from pathlib import Path

HEIGHT = 6
WIDTH = 8
# Nonlexical FOV numbers with printed pixel origins (x, y).
POSITIONS = {2: (1000.0, 5000.5), 10: (3000.25, 7000.0)}
# Single-pixel cells at (row, column): their local centroid is exactly (column, HEIGHT - row).
LANDMARKS = {1: (1, 2), 2: (4, 6)}


@dataclass(frozen=True, slots=True)
class CosmxFixture:
    """Identify one written export and its scientific oracles."""

    root: Path
    dataset_id: str
    profile: Literal["flat_2021", "atomx"]
    features: tuple[str, ...]


def local_xy(cell: int) -> tuple[float, float]:
    """Return the evidenced local coordinates of a landmark cell."""
    row, column = LANDMARKS[cell]
    return float(column), float(HEIGHT - row)


def global_xy(fov: int, cell: int, profile: Literal["flat_2021", "atomx"]) -> tuple[float, float]:
    """Return global coordinates: positive axes in 2021 exports, a reversed y axis in AtoMx."""
    x, y = local_xy(cell)
    origin_x, origin_y = POSITIONS[fov]
    return origin_x + x, (origin_y + y) if profile == "flat_2021" else (origin_y - y)


def _write_csv(path: Path, frame: pd.DataFrame, *, compressed: bool) -> None:
    """Write plain CSV or fwrite-like gzip: a header member, then one member per row block."""
    if not compressed:
        frame.to_csv(path, index=False)
        return
    header = frame.iloc[:0].to_csv(index=False).encode()
    with path.open("wb") as handle:
        handle.write(gzip.compress(header, mtime=0))
        for start in range(0, len(frame), 2):
            handle.write(
                gzip.compress(frame.iloc[start : start + 2].to_csv(index=False, header=False).encode(), mtime=0)
            )


def _metadata(profile: Literal["flat_2021", "atomx"]) -> pd.DataFrame:
    """Return landmark cell metadata whose centers follow the evidenced conventions."""
    rows = []
    for fov in POSITIONS:
        for cell in LANDMARKS:
            x, y = local_xy(cell)
            gx, gy = global_xy(fov, cell, profile)
            rows.append(
                {
                    "fov": fov,
                    "cell_ID": cell,
                    "Area": 1,
                    "AspectRatio": 1.0,
                    "CenterX_local_px": x,
                    "CenterY_local_px": y,
                    "CenterX_global_px": gx,
                    "CenterY_global_px": gy,
                    "Mean.DAPI": "" if (fov, cell) == (10, 2) else 10 * fov + cell,
                    "note": f"0{cell}",
                }
            )
    metadata = pd.DataFrame(rows)
    if profile == "atomx":
        metadata.insert(0, "cell", [f"c_1_{fov}_{cell}" for fov in POSITIONS for cell in LANDMARKS])
        metadata["cell_id"] = metadata["cell"]
        metadata["slide_ID"] = 1
    return metadata


def _counts(features: tuple[str, ...], controls: int) -> pd.DataFrame:
    """Return count rows deliberately reordered against metadata, with per-FOV background rows."""
    rows = [
        (10, 2, 0, 0, 7, *([1] * controls)),
        (2, 0, 9, 9, 9, *([9] * controls)),
        (2, 2, 0, 3, 0, *([0] * controls)),
        (10, 1, 4, 0, 0, *([0] * controls)),
        (10, 0, 9, 9, 9, *([9] * controls)),
        (2, 1, 1, 0, 2, *([0] * controls)),
    ]
    return pd.DataFrame(rows, columns=["fov", "cell_ID", *features])


def _positions(profile: Literal["flat_2021", "atomx"]) -> pd.DataFrame:
    """Return pixel positions for 2021 exports and millimetre-only positions for AtoMx."""
    if profile == "atomx":
        return pd.DataFrame({"Slide": 1, "X_mm": [1.0, 2.0], "Y_mm": [3.0, 4.0], "FOV": list(POSITIONS)})
    origins = list(POSITIONS.values())
    return pd.DataFrame({"fov": list(POSITIONS), "x_px": [o[0] for o in origins], "y_px": [o[1] for o in origins]})


def _transcripts(profile: Literal["flat_2021", "atomx"], control: str) -> pd.DataFrame:
    """Return assigned and unassigned events with sub-pixel local and exact global coordinates."""
    rows = []
    for fov, (origin_x, origin_y) in POSITIONS.items():
        for index, (cell, target, z) in enumerate(((1, "GeneA", 0), (0, "HLA-A", -1), (2, control, 2))):
            x, y = local_xy(cell or 1)
            x, y = x + 0.25 * index, y + 0.125
            row = {
                "fov": fov,
                "cell_ID": cell,
                "x_global_px": origin_x + x,
                "y_global_px": origin_y + y if profile == "flat_2021" else origin_y - y,
                "x_local_px": x,
                "y_local_px": y,
                "z": z,
                "target": target,
                "CellComp": "Nuclear" if cell else ("0" if profile == "flat_2021" else "None"),
            }
            if profile == "atomx":
                row["cell"] = f"c_1_{fov}_{cell}" if cell else ""
            rows.append(row)
    return pd.DataFrame(rows)


def _write_rasters(path: Path) -> None:
    """Write lossless composites and labels whose landmark pixels identify each cell."""
    (path / "CellComposite").mkdir(exist_ok=True)
    (path / "CellLabels").mkdir(exist_ok=True)
    for fov in POSITIONS:
        labels = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
        for cell, (row, column) in LANDMARKS.items():
            labels[row, column] = cell
        tifffile.imwrite(path / "CellLabels" / f"CellLabels_F{fov:03d}.tif", labels, photometric="minisblack")
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        image[..., 0] = np.arange(HEIGHT * WIDTH, dtype=np.uint8).reshape(HEIGHT, WIDTH)
        image[..., 1] = fov
        (path / "CellComposite" / f"CellComposite_F{fov:03d}.png").write_bytes(imagecodecs.png_encode(image))


def write_cosmx_dataset(  # noqa: PLR0913, RUF100 - Independent format variants exercise reader boundaries.
    path: Path,
    *,
    profile: Literal["flat_2021", "atomx"] = "flat_2021",
    compressed: bool = False,
    rasters: bool = True,
    positions: bool = True,
    dataset_id: str = "Run_1",
) -> CosmxFixture:
    """Write counts, metadata, positions, rasters, and transcripts for FOVs 2 and 10."""
    path.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compressed else ".csv"
    controls = ("NegPrb1",) if profile == "flat_2021" else ("Negative1", "SystemControl1")
    features = ("GeneB", "HLA-A", "GeneA", *controls)

    _write_csv(path / f"{dataset_id}_metadata_file{suffix}", _metadata(profile), compressed=compressed)
    _write_csv(path / f"{dataset_id}_exprMat_file{suffix}", _counts(features, len(controls)), compressed=compressed)
    if positions:
        _write_csv(path / f"{dataset_id}_fov_positions_file{suffix}", _positions(profile), compressed=compressed)
    _write_csv(path / f"{dataset_id}_tx_file{suffix}", _transcripts(profile, controls[0]), compressed=compressed)
    if rasters:
        _write_rasters(path)
    return CosmxFixture(path, dataset_id, profile, features)
