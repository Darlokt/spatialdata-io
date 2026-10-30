"""Write deterministic, format-faithful Spatial Genomics ROI artifacts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tifffile

if TYPE_CHECKING:
    from pathlib import Path


def write_seqfish_dataset(  # noqa: PLR0913, RUF100 - Fixture source variations are independent.
    root: Path,
    *,
    rois: tuple[str, ...] = ("Roi2", "Roi1"),
    cell_ids: tuple[int, int] = (9, 2),
    physical_size_x: float = 0.5,
    physical_size_y: float = 0.25,
    with_z: bool = True,
    with_cell: bool = True,
    reorder_cells: bool = False,
    reorder_boundaries: bool = False,
    singleton_axes: bool = False,
    invalid_cell_ids: tuple[int, ...] = (),
) -> Path:
    """Write two-cell ROI layouts using the evidenced source headers and IDs.

    Counts and cell rows deliberately preserve unsorted IDs. Boundaries are in
    pixel coordinates; centroids and transcript x/y are in physical ROI units.
    Transcript z is an integer source slice annotation, with no asserted unit.
    """
    root.mkdir(parents=True, exist_ok=True)
    for roi in rois:
        pd.DataFrame({"GeneB": [3, 0], "GeneA": [0, 5]}, index=pd.Index(cell_ids, name="")).to_csv(
            root / f"{roi}_CellxGene.csv"
        )
        cells = pd.DataFrame(
            {"label": cell_ids, "area": [np.pi, 4 * np.pi], "center_x": [1.5, 4.5], "center_y": [0.5, 1.25]}
        )
        cells.iloc[:: -1 if reorder_cells else 1].to_csv(root / f"{roi}_CellCoordinates.csv", index=False)
        dapi = np.arange(96, dtype=np.uint16).reshape(8, 12)
        metadata = {
            "axes": "CYX" if singleton_axes else "YX",
            "PhysicalSizeX": physical_size_x,
            "PhysicalSizeY": physical_size_y,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        }
        tifffile.imwrite(
            root / f"{roi}_DAPI.tiff",
            dapi[np.newaxis] if singleton_axes else dapi,
            ome=True,
            metadata=metadata,
            rowsperstrip=2,
            compression="deflate",
        )
        labels = np.zeros((8, 12), dtype=np.uint32)
        labels[1:3, 2:5] = cell_ids[0]
        labels[4:7, 8:11] = cell_ids[1]
        tifffile.imwrite(root / f"{roi}_Segmentation.tiff", labels, rowsperstrip=2, compression="deflate")
        transcripts = pd.DataFrame({"name": ["GeneA", "GeneB", "GeneA"], "x": [4.5, 1.5, 4.75], "y": [1.25, 0.5, 1.5]})
        if with_z:
            transcripts["z"] = [1, 2, 1]
        if with_cell:
            transcripts["cell"] = [cell_ids[1], cell_ids[0], 0]
        transcripts.to_csv(root / f"{roi}_TranscriptList.csv", index=False)
        features = [
            {
                "type": "Feature",
                "id": str(identifier),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[2, 1], [5, 3], [5, 1], [2, 3], [2, 1]] if identifier in invalid_cell_ids else ring
                    ],
                },
                "properties": {"objectType": "cell"},
            }
            for identifier, ring in zip(
                cell_ids,
                (
                    [[2, 1], [5, 1], [5, 3], [2, 3], [2, 1]],
                    [[8, 4], [11, 4], [11, 7], [8, 7], [8, 4]],
                ),
                strict=True,
            )
        ]
        if reorder_boundaries:
            features.reverse()
        (root / f"{roi}_Boundaries.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
        )
    return root
