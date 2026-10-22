"""Small processor-style CODEX CSV, FCS 3.1, and stitched-image artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tifffile

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def cell_frame() -> pd.DataFrame:
    """Return numeric processor fields in deliberately non-sorted source order."""
    return pd.DataFrame(
        {
            "cell_id": [7, 2, 11],
            "region": [1, 1, 1],
            "tile_num": [2, 1, 2],
            "x": [20.0, 2.5, 13.0],
            "y": [1.5, 30.0, 8.0],
            "z": [1, 1, 1],
            "x_tile": [4.0, 2.5, 13.0],
            "y_tile": [1.5, 14.0, 8.0],
            "size": [5, 6, 7],
            "cyc002_ch001": [0.0, 3.25, -1.5],
            "cyc001_ch002": [2.0, 0.0, 5.5],
            "tsne_x": [0.25, 0.5, 0.75],
            "tsne_y": [0.75, 0.5, 0.25],
            "homogeneity": [1.0, 1.25, 1.5],
        }
    )


def write_csv(path: Path, frame: pd.DataFrame | None = None) -> None:
    """Write the processor's equivalent CSV column schema."""
    (cell_frame() if frame is None else frame).to_csv(path, index=False)


def write_fcs(path: Path, *, empty: bool = False, marker_labels: bool = True) -> None:
    """Write actual FCS 3.1 list-mode little-endian float32 events.

    Channel names use the vendor PnN keys; PnS carries human marker labels.
    Fixed eight-character offsets keep TEXT and HEADER offsets identical.
    This writer requires no extra test dependency and is checked by readfcs.
    """
    frame = cell_frame().iloc[:0] if empty else cell_frame()
    events = frame.to_numpy(dtype="<f4").tobytes()
    text_start = 256
    keywords = {
        "$BEGINANALYSIS": "0",
        "$ENDANALYSIS": "0",
        "$BEGINSTEXT": "0",
        "$ENDSTEXT": "0",
        "$BEGINDATA": "00000000",
        "$ENDDATA": "00000000",
        "$BYTEORD": "1,2,3,4",
        "$DATATYPE": "F",
        "$MODE": "L",
        "$NEXTDATA": "0",
        "$PAR": str(len(frame.columns)),
        "$TOT": str(len(frame)),
    }
    labels = {"cyc002_ch001": "CD45", "cyc001_ch002": "DNA"}
    for number, column in enumerate(frame.columns, start=1):
        prefix = f"$P{number}"
        keywords.update(
            {
                f"{prefix}B": "32",
                f"{prefix}E": "0,0",
                f"{prefix}G": "1",
                f"{prefix}R": "262144",
                f"{prefix}N": column,
                **({f"{prefix}S": labels.get(column, column)} if marker_labels else {}),
            }
        )
    text = "/" + "/".join(part for pair in keywords.items() for part in pair) + "/"
    data_start = text_start + len(text)
    data_end = data_start + len(events) - 1
    keywords["$BEGINDATA"] = f"{data_start:08d}"
    keywords["$ENDDATA"] = f"{data_end:08d}"
    text = "/" + "/".join(part for pair in keywords.items() for part in pair) + "/"
    offsets = (text_start, data_start - 1, data_start, data_end, 0, 0)
    header = "FCS3.1    " + "".join(f"{offset:8d}" for offset in offsets)
    path.write_bytes(header.encode("ascii").ljust(text_start, b" ") + text.encode("ascii") + events)


def write_image(path: Path, *, compressed: bool = False) -> NDArray[np.uint16]:
    """Write two stitched marker channels with explicit CYX metadata."""
    image = np.arange(2 * 32 * 40, dtype=np.uint16).reshape(2, 32, 40)
    tifffile.imwrite(
        path,
        image,
        photometric="minisblack",
        metadata={"axes": "CYX"},
        compression="deflate" if compressed else None,
        tile=(16, 16),
    )
    return image
