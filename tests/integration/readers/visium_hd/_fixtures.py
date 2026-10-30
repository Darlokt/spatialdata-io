"""Deterministic tiny Space Ranger structures with independently defined count truth.

These fixtures model documented 3.x and 4.x artifact contracts, not producer runs.
The 17-square base grid exercises partial 8/16 µm edges. Filtered matrix order
is intentionally spatially unsorted, and an out-of-tissue nucleus bin has counts.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
from scipy.sparse import csr_array

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

GRID_SHAPE = (17, 17)
MICRONS_PER_PIXEL = 0.5
HOMOGRAPHY = np.array([[4.0, 0.03, 4.0], [0.02, -4.0, 80.0], [0.00001, -0.00002, 1.0]])
FEATURE_IDS = ("gene-a", "gene-b", "gene-c")
FEATURE_NAMES = ("Gene", "Gene", "Other")
GENOMES = ("mouse", "human", "mouse")
CELL_IDS = ("cellid_000000001-1", "cellid_000000002-1", "cellid_000000003-1")


@dataclass(frozen=True, slots=True)
class Bundle:
    """Paths and exact tiny source matrices, in their on-disk observation order."""

    root: Path
    version: str
    sizes: tuple[int, ...]
    raw: dict[int, NDArray[np.int32]]
    filtered_rows: dict[int, NDArray[np.int64]]
    cells: NDArray[np.int32]
    nuclei: NDArray[np.int32]


def barcode(size: int, row: int, col: int) -> str:
    """Return the documented square barcode spelling."""
    return f"s_{size:03d}um_{row:05d}_{col:05d}-1"


def write_counts(path: Path, values: NDArray[np.int32], barcodes: list[str], version: str) -> None:
    """Write a v3 10x HDF5 matrix using native sparse arrays and byte strings."""
    matrix = csr_array(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["software_version"] = f"spaceranger-{version}"
        handle.attrs["filetype"] = "matrix"
        handle.attrs["version"] = 2
        group = handle.create_group("matrix")
        group.create_dataset("data", data=matrix.data.astype(np.int32))
        group.create_dataset("indices", data=matrix.indices.astype(np.int64))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
        group.create_dataset("shape", data=np.array(values.T.shape, dtype=np.int32))
        group.create_dataset("barcodes", data=np.asarray(barcodes, dtype="S45"))
        features = group.create_group("features")
        for name, content in (
            ("id", FEATURE_IDS),
            ("name", FEATURE_NAMES),
            ("genome", GENOMES),
            ("feature_type", ("Gene Expression",) * 3),
            ("_all_tag_keys", ("genome",)),
        ):
            features.create_dataset(name, data=np.asarray(content, dtype="S32"))


def _pixels(col: NDArray[np.float64], row: NDArray[np.float64]) -> NDArray[np.float64]:
    homogeneous = np.column_stack((col, row, np.ones(col.size))) @ HOMOGRAPHY.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def write_bundle(  # noqa: PLR0913, RUF100 - Independently select supported fixture products.
    root: Path,
    *,
    version: str = "4.1.0",
    sizes: tuple[int, ...] = (2, 8, 16),
    segmented: bool = True,
    prefix: str = "",
    feature_slice_name: Literal["feature_slice.h5", "hd_feature_slice.h5"] = "feature_slice.h5",
    null_pair: Literal["microscope", "cytassist"] | None = None,
    images: bool = False,
    fullres: Literal["root", "folder"] | None = None,
) -> Bundle:
    """Write bins, optional segmentation, mappings and calibrated RGB images."""
    root.mkdir(parents=True, exist_ok=True)
    base = np.zeros((17 * 17, 3), dtype=np.int32)
    base[0] = [1, 2, 0]
    base[1] = [0, 3, 0]
    base[17] = [0, 0, 4]
    base[20] = [2, 0, 1]
    scale = {
        "microns_per_pixel": MICRONS_PER_PIXEL,
        "tissue_hires_scalef": 0.5,
        "tissue_lowres_scalef": 0.25,
        "fiducial_diameter_fullres": 660.0,
        "regist_target_img_scalef": 1.0,
    }
    raw, selected = _write_bins(root, version, sizes, base, scale)
    matrices: dict[str, list[float] | None] = {}
    for frame, matrix in (
        ("microscope", HOMOGRAPHY),
        ("cytassist", np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]])),
    ):
        matrices[f"spot_colrow_to_{frame}_colrow"] = None if frame == null_pair else matrix.ravel().tolist()
        matrices[f"{frame}_colrow_to_spot_colrow"] = (
            None if frame == null_pair else np.linalg.inv(matrix).ravel().tolist()
        )
    metadata = {
        "sample_id": "synthetic",
        "nrows": 17,
        "ncols": 17,
        "spot_pitch": 2.0,
        "hd_layout_json": json.dumps({"file_format": "1.0", "transform": np.eye(3).ravel().tolist()}),
        "transform_matrices": matrices,
    }
    with h5py.File(root / f"{prefix}{feature_slice_name}", "w") as handle:
        handle.attrs["software_version"] = f"spaceranger-{version}"
        handle.attrs["filetype"] = "feature_slice"
        handle.attrs["metadata_json"] = json.dumps(metadata)
    # Cell 1 includes base bins 0 and 1; cell 2 includes 17 and 20; cell 3 is empty.
    cells = np.array([base[0] + base[1], base[17] + base[20], [0, 0, 0]], dtype=np.int32)
    nuclei = np.array([base[1], base[17], [0, 0, 0]], dtype=np.int32)
    if segmented:
        directory = root / "segmented_outputs"
        spatial = directory / "spatial"
        spatial.mkdir(parents=True)
        (spatial / "scalefactors_json.json").write_text(json.dumps(scale))
        write_counts(directory / "raw_feature_cell_matrix.h5", cells, list(CELL_IDS), version)
        write_counts(directory / "filtered_feature_cell_matrix.h5", cells[[1, 0]], [CELL_IDS[1], CELL_IDS[0]], version)
        for kind, half in (("cell", 3.0), ("nucleus", 1.0)):
            features = []
            for cell in range(1, 4):
                x, y = 10.0 * cell, 20.0
                ring = [
                    [x - half, y - half],
                    [x + half, y - half],
                    [x + half, y + half],
                    [x - half, y + half],
                    [x - half, y - half],
                ]
                properties: dict[str, int | list[float]] = {"cell_id": cell}
                if version.startswith("4.1"):
                    properties.update({"cell_centroid": [x, y], "nucleus_centroid": [x, y]})
                features.append(
                    {
                        "type": "Feature",
                        "properties": properties,
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    }
                )
            (directory / f"{kind}_segmentations.geojson").write_text(
                json.dumps({"type": "FeatureCollection", "features": features})
            )
        mapping: dict[str, list[str | bool | None]] = {
            f"square_{size:03d}um": [
                barcode(size, r // (size // 2), c // (size // 2)) for r in range(17) for c in range(17)
            ]
            for size in (2, 8, 16)
        }
        mapping["cell_id"] = [
            CELL_IDS[0] if i in (0, 1) else CELL_IDS[1] if i in (17, 20) else None for i in range(289)
        ]
        mapping["in_cell"] = [i in (0, 1, 17, 20) for i in range(289)]
        mapping["in_nucleus"] = [i in (1, 17) for i in range(289)]
        pq.write_table(pa.table(mapping), directory / "barcode_mappings.parquet", row_group_size=37)
    if images:
        for spatial in sorted(root.glob("*/spatial")) + sorted(root.glob("binned_outputs/*/spatial")):
            _write_png(spatial / "tissue_hires_image.png", np.arange(48 * 48 * 3, dtype=np.uint8).reshape(48, 48, 3))
            _write_png(spatial / "tissue_lowres_image.png", np.zeros((24, 24, 3), dtype=np.uint8))
            tifffile.imwrite(spatial / "cytassist_image.tiff", np.ones((48, 48, 3), dtype=np.uint8), photometric="rgb")
    if fullres is not None:
        destination = (
            root / f"{prefix}tissue_image.tif" if fullres == "root" else root / "microscope_image" / "image.tif"
        )
        destination.parent.mkdir(exist_ok=True)
        tifffile.imwrite(destination, np.arange(96 * 96 * 3, dtype=np.uint16).reshape(96, 96, 3), photometric="rgb")
    return Bundle(root, version, sizes, raw, selected, cells, nuclei)


def _write_png(path: Path, pixels: NDArray[np.uint8]) -> None:
    # A minimal RGB PNG writer keeps image fixtures independent of Pillow.
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))

    height, width, _ = pixels.shape
    rows = b"".join(b"\0" + row.tobytes() for row in pixels)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _write_bins(
    root: Path,
    version: str,
    sizes: tuple[int, ...],
    base: NDArray[np.int32],
    scale: dict[str, float],
) -> tuple[dict[int, NDArray[np.int32]], dict[int, NDArray[np.int64]]]:
    raw: dict[int, NDArray[np.int32]] = {}
    selected: dict[int, NDArray[np.int64]] = {}
    for size in sizes:
        factor = size // 2
        nrows = (17 + factor - 1) // factor
        rows, cols = np.indices((nrows, nrows), dtype=np.uint32)
        rows, cols = rows.ravel(), cols.ravel()
        values = np.zeros((nrows * nrows, 3), dtype=np.int32)
        for i in range(17 * 17):
            row, col = divmod(i, 17)
            values[(row // factor) * nrows + col // factor] += base[i]
        indices = np.array(
            list(dict.fromkeys([min(20, len(values) - 1), 0, min(17, len(values) - 1), 5])), dtype=np.int64
        )
        raw[size], selected[size] = values, indices
        directory = root / "binned_outputs" / f"square_{size:03d}um"
        spatial = directory / "spatial"
        spatial.mkdir(parents=True)
        names = [barcode(size, int(r), int(c)) for r, c in zip(rows, cols, strict=True)]
        write_counts(directory / "raw_feature_bc_matrix.h5", values, names, version)
        write_counts(directory / "filtered_feature_bc_matrix.h5", values[indices], [names[i] for i in indices], version)
        # Space Ranger uses one image-space offset for every bin, including
        # partial edges, rather than projecting each nominal bin centre.
        corners = _pixels(np.array([0.0, factor - 1.0]), np.array([0.0, factor - 1.0]))
        pixels = _pixels(cols.astype(np.float64) * factor, rows.astype(np.float64) * factor)
        pixels += (corners[1] - corners[0]) / 2
        tissue = np.zeros(len(values), dtype=np.uint8)
        tissue[indices] = 1
        schema = pa.schema(
            [
                pa.field("barcode", pa.string(), nullable=False),
                pa.field("in_tissue", pa.uint8(), nullable=False),
                pa.field("array_row", pa.uint32(), nullable=False),
                pa.field("array_col", pa.uint32(), nullable=False),
                pa.field("pxl_row_in_fullres", pa.float64(), nullable=False),
                pa.field("pxl_col_in_fullres", pa.float64(), nullable=False),
            ]
        )
        table = pa.Table.from_arrays(
            [
                pa.array(names),
                pa.array(tissue),
                pa.array(rows),
                pa.array(cols),
                pa.array(pixels[:, 1]),
                pa.array(pixels[:, 0]),
            ],
            schema=schema,
        )
        pq.write_table(table, spatial / "tissue_positions.parquet", row_group_size=37)
        (spatial / "scalefactors_json.json").write_text(
            json.dumps({**scale, "bin_size_um": size, "spot_diameter_fullres": size / MICRONS_PER_PIXEL})
        )
    return raw, selected
