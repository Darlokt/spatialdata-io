"""Write deterministic synthetic Xenium bundles; no fixture claims an observed producer run.

Every bundle derives from one small geometric truth, so each expected output is
known independently of the reader:

- masks are rectangles on a 24 x 32 pixel grid with a 0.2125 µm pixel size;
- boundary polygons trace boundary-pixel centres in pixel-index coordinates,
  the vendor convention measured on real bundles (vertices sit half a pixel up
  and left of the corner frame shared by masks, centroids and transcripts);
- centroids and areas follow the mask pixels in the corner frame;
- count matrices hold the assigned transcripts with ``qv >= 20``;
- protein intensities are stored scaled by ten.

Structures follow the documented and observed format families: integer cell IDs
(XOA 1.0-1.2), hexadecimal IDs with ``seg_mask_value`` (XOA 1.3-1.9) and polygon
sets (XOA 2.0 onwards, Xenium Ranger outputs). Cell-ID strings are written
literally from the documented 10x codec, never from production code.
"""

from __future__ import annotations

import gzip
import io
import json
import shutil
import tarfile
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import tifffile
import zarr
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from zarr.core.common import JSON

type Family = Literal["integer_ids", "hex_ids", "polygon_sets"]
type Generation = Literal["1.0", "1.3", "1.5", "2.0", "3.0", "4.0"]
type ImageNaming = Literal["single", "numbered", "named"]
type TableFormat = Literal["parquet", "csv", "csv_gz"]
type CountFormat = Literal["h5", "mtx", "mtx_archive"]

PIXEL_SIZE = 0.2125
SHAPE = (24, 32)
MASK_CHUNKS = (8, 8)
Z_STEP = 3.0
SEED = 20261026


@dataclass(frozen=True, slots=True)
class Rect:
    """An inclusive rectangle of mask pixels."""

    top: int
    left: int
    bottom: int
    right: int

    @property
    def pixels(self) -> int:
        """Return the number of covered pixels."""
        return (self.bottom - self.top + 1) * (self.right - self.left + 1)

    def contains(self, row: int, column: int) -> bool:
        """Return whether a pixel index lies in the rectangle."""
        return self.top <= row <= self.bottom and self.left <= column <= self.right

    def ring(self, *, bowtie: bool = False) -> list[tuple[float, float]]:
        """Return the closed micrometre ring through boundary-pixel centres in index coordinates."""
        corners = [(self.left, self.top), (self.right, self.top), (self.right, self.bottom), (self.left, self.bottom)]
        if bowtie:
            corners = [corners[0], corners[2], corners[1], corners[3]]
        ring = [(x * PIXEL_SIZE, y * PIXEL_SIZE) for x, y in corners]
        return [*ring, ring[0]]


CELL_RECTS = (Rect(2, 2, 7, 9), Rect(2, 12, 7, 19), Rect(12, 2, 19, 13), Rect(12, 18, 19, 27))
# Strict 1:1 nuclei of the integer and hexadecimal families.
NUCLEI_ONE_TO_ONE = ((0, Rect(4, 4, 5, 6)), (1, Rect(4, 14, 5, 16)), (2, Rect(14, 4, 17, 8)), (3, Rect(14, 20, 17, 24)))
# Polygon sets: cell 2 has two nuclei, cell 3 none.
NUCLEI_POLYGON_SETS = (
    (0, Rect(4, 4, 5, 6)),
    (1, Rect(4, 14, 5, 16)),
    (2, Rect(14, 3, 17, 6)),
    (2, Rect(14, 9, 17, 12)),
)

# The documented codec: eight hexadecimal digits of the uint32 prefix, each digit
# 0-f mapped to a-p, then "-" and the suffix. 1437536272 is the documented example.
HEX_PAIRS = ((1, 1), (1437536272, 1), (2147483653, 1), (4294967295, 1))
HEX_IDS = ("aaaaaaab-1", "ffkpbaba-1", "iaaaaaaf-1", "pppppppp-1")
INTEGER_IDS = (1, 2, 3, 4)

SEGMENTATION_METHODS = (
    "Segmented by boundary stain (ATP1A1+CD45+E-Cadherin)",
    "Segmented by interior stain (18S)",
    "Segmented by nucleus expansion of 5.0um",
    "Segmented by nuclear stain (DAPI)",
    "Imported Cell Segmentation",
)
CELL_METHODS = (0, 1, 2, 2)
IMPORTED_METHOD = 4

GENES = (("ENSG00000000001", "GeneA"), ("ENSG00000000002", "GeneB"), ("ENSG00000000003", "GeneC"))
PROTEINS = (("TXP000019", "PD-1"), ("TXP000023", "VISTA"))
PROTEIN_STORED = ((7, 0), (12, 3), (0, 0), (629, 5))
PROTEIN_FACTOR = 10

FOCUS_FOUR = ("DAPI", "ATP1A1/CD45/E-Cadherin", "18S", "alphaSMA/Vimentin")
FOCUS_PROTEIN = (
    *FOCUS_FOUR,
    "Blue background",
    "Green background",
    "Yellow background",
    "Red background",
    "PD-1",
    "VISTA",
)


@dataclass(frozen=True, slots=True)
class Profile:
    """One documented or observed bundle generation."""

    family: Family
    generation: Generation
    version: tuple[int, int, int | None] | None
    software: str
    images: ImageNaming
    channels: tuple[str, ...]
    protein: bool = False
    binary_ids: bool = False


XOA_1_0 = Profile("integer_ids", "1.0", None, "Xenium-1.0.1", "single", ("DAPI",))
XOA_1_3 = Profile("hex_ids", "1.3", (2, 0, None), "xenium-1.3.0.5", "single", ("DAPI",), binary_ids=True)
XOA_1_9 = Profile("hex_ids", "1.5", (3, 0, 0), "xenium-1.9.0.1", "single", ("DAPI",))
XOA_2_0 = Profile("polygon_sets", "2.0", (4, 0, 0), "xenium-2.0.0.10", "numbered", FOCUS_FOUR)
XOA_3_0 = Profile("polygon_sets", "3.0", (5, 0, 0), "xenium-3.0.0.15", "numbered", ("DAPI",))
XOA_4_0 = Profile("polygon_sets", "4.0", (6, 0, 0), "xenium-4.0.0.19", "named", FOCUS_PROTEIN, protein=True)
FORMAT_6_1 = Profile("polygon_sets", "4.0", (6, 1, 0), "xenium-4.0.2.4", "named", ("DAPI",))


@dataclass(frozen=True, slots=True)
class Options:
    """Artifact and defect choices for one written bundle."""

    tables: TableFormat = "parquet"
    csv_twins: bool = False
    counts: tuple[CountFormat, ...] = ("h5",)
    transcripts: TableFormat | None = "parquet"
    transcript_row_group: int | None = None
    cells_store: bool = True
    images: bool = True
    mip: bool = True
    nucleus_label_id: bool = True
    ranger: Literal["relabel", "rename"] | None = None
    import_kind: Literal["transcripts", "cells_only"] | None = None
    bowtie_cells: tuple[int, ...] = ()
    bowtie_nuclei: tuple[int, ...] = ()
    short_nuclei: tuple[int, ...] = ()
    aligned: tuple[Literal["he", "if"], ...] = ()
    analysis: Literal["archive", "directory"] | None = None
    metadata: bool = True


@dataclass(frozen=True, slots=True)
class Transcripts:
    """Source transcript records in file order."""

    transcript_id: NDArray[np.uint64]
    cell_index: NDArray[np.int64]
    feature_index: NDArray[np.int64]
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    z: NDArray[np.float32]
    qv: NDArray[np.float32]
    overlaps_nucleus: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class XeniumFixture:
    """Paths and independently known truth of one written bundle."""

    root: Path
    profile: Profile
    options: Options
    cell_ids: tuple[str, ...] | tuple[int, ...]
    barcodes: tuple[str, ...]
    cell_labels: tuple[int, ...]
    nuclei: tuple[tuple[int, Rect], ...]
    nucleus_labels: tuple[int, ...]
    features: tuple[str, ...]
    feature_ids: tuple[str, ...]
    feature_types: tuple[str, ...]
    counts: NDArray[np.int32]
    transcripts: Transcripts
    cell_mask: NDArray[np.uint32]
    nucleus_mask: NDArray[np.uint32]
    focus: NDArray[np.uint16]
    focus_entry: str
    channels: tuple[str, ...]
    experiment: dict[str, object] = field(default_factory=dict)


def _features(profile: Profile) -> tuple[tuple[str, str, str], ...]:
    """Return (id, name, type) in matrix order for one generation."""
    unassigned = (
        ("BLANK_0001", "Blank Codeword")
        if profile.generation == "1.0"
        else (
            "UnassignedCodeword_0001",
            "Unassigned Codeword",
        )
    )
    features = [(gene_id, name, "Gene Expression") for gene_id, name in GENES]
    features += [
        ("NegControlProbe_00001", "NegControlProbe_00001", "Negative Control Probe"),
        ("NegControlCodeword_0500", "NegControlCodeword_0500", "Negative Control Codeword"),
        (unassigned[0], unassigned[0], unassigned[1]),
    ]
    if profile.generation in ("3.0", "4.0"):
        features += [
            ("Intergenic_Region_1", "Intergenic_Region_1", "Genomic Control"),
            ("DeprecatedCodeword_0001", "DeprecatedCodeword_0001", "Deprecated Codeword"),
        ]
    if profile.protein:
        features += [(protein_id, name, "Protein Expression") for protein_id, name in PROTEINS]
    return tuple(features)


def _nuclei(profile: Profile, options: Options) -> tuple[tuple[int, Rect], ...]:
    if options.import_kind == "cells_only":
        return ()
    return NUCLEI_POLYGON_SETS if profile.family == "polygon_sets" else NUCLEI_ONE_TO_ONE


def _masks(nuclei: tuple[tuple[int, Rect], ...], nucleus_labels: tuple[int, ...]) -> tuple[NDArray[np.uint32], ...]:
    cell_mask = np.zeros(SHAPE, dtype=np.uint32)
    for index, rect in enumerate(CELL_RECTS):
        cell_mask[rect.top : rect.bottom + 1, rect.left : rect.right + 1] = index + 1
    nucleus_mask = np.zeros(SHAPE, dtype=np.uint32)
    for label, (_, rect) in zip(nucleus_labels, nuclei, strict=True):
        nucleus_mask[rect.top : rect.bottom + 1, rect.left : rect.right + 1] = label
    return cell_mask, nucleus_mask


def _transcripts(nuclei: tuple[tuple[int, Rect], ...], n_features: int) -> Transcripts:
    """Place six transcripts in every cell and four unassigned ones between cells."""
    rng = np.random.default_rng(SEED)
    rows: list[tuple[int, int, int, int, float]] = []
    for cell, rect in enumerate(CELL_RECTS):
        for _ in range(6):
            row = int(rng.integers(rect.top, rect.bottom + 1))
            column = int(rng.integers(rect.left, rect.right + 1))
            rows.append(
                (cell, row, column, int(rng.integers(0, n_features)), float(rng.choice([12.5, 20.0, 31.25, 40.0])))
            )
    rows.extend((-1, 10, column, int(rng.integers(0, n_features)), 40.0) for column in (0, 11, 16, 30))
    rng.shuffle(rows)

    cell_index = np.array([row[0] for row in rows], dtype=np.int64)
    pixel_rows = np.array([row[1] for row in rows])
    pixel_columns = np.array([row[2] for row in rows])
    # Fractions in [0.25, 0.75] keep floor(x / pixel_size) on the assigned pixel.
    fractions = rng.uniform(0.25, 0.75, size=(len(rows), 2))
    overlaps = [
        any(owner == cell and rect.contains(r, c) for owner, rect in nuclei)
        for cell, r, c in zip(cell_index, pixel_rows, pixel_columns, strict=True)
    ]
    return Transcripts(
        transcript_id=np.arange(281_552_286_122_600, 281_552_286_122_600 + len(rows), dtype=np.uint64),
        cell_index=cell_index,
        feature_index=np.array([row[3] for row in rows], dtype=np.int64),
        x=((pixel_columns + fractions[:, 0]) * PIXEL_SIZE).astype(np.float32),
        y=((pixel_rows + fractions[:, 1]) * PIXEL_SIZE).astype(np.float32),
        z=(np.round(rng.uniform(8.0, 20.0, size=len(rows)) * 64) / 64).astype(np.float32),
        qv=np.array([row[4] for row in rows], dtype=np.float32),
        overlaps_nucleus=np.array(overlaps, dtype=np.uint8),
    )


def _counts(transcripts: Transcripts, features: tuple[tuple[str, str, str], ...]) -> NDArray[np.int32]:
    """Count assigned q20 transcripts per cell and feature; proteins are stored scaled values."""
    counts = np.zeros((len(CELL_RECTS), len(features)), dtype=np.int32)
    assigned = (transcripts.cell_index >= 0) & (transcripts.qv >= 20)
    np.add.at(counts, (transcripts.cell_index[assigned], transcripts.feature_index[assigned]), 1)
    protein = [index for index, feature in enumerate(features) if feature[2] == "Protein Expression"]
    if protein:
        counts[:, protein] = np.asarray(PROTEIN_STORED, dtype=np.int32)
    return counts


def _ids(profile: Profile) -> tuple[str, ...] | tuple[int, ...]:
    return INTEGER_IDS if profile.family == "integer_ids" else HEX_IDS


def _id_array(profile: Profile, values: Sequence[str | int]) -> pa.Array:
    if profile.family == "integer_ids":
        return pa.array(values, type=pa.int32())
    if profile.binary_ids:
        return pa.array([str(value).encode() for value in values], type=pa.binary())
    return pa.array([str(value) for value in values], type=pa.string())


def _write_table(table: pa.Table, root: Path, stem: str, *, kind: TableFormat, twins: bool) -> None:
    """Write a parquet file or its CSV twin, with full-precision text numbers."""
    kinds: list[TableFormat] = [kind]
    if twins and kind == "parquet":
        kinds.append("csv_gz")
    for current in kinds:
        if current == "parquet":
            pq.write_table(table, root / f"{stem}.parquet")
            continue
        text = table
        if pa.types.is_binary(table.schema.field("cell_id").type):
            text = table.set_column(0, "cell_id", table.column("cell_id").cast(pa.string()))
        buffer = io.BytesIO()
        pacsv.write_csv(text, buffer, pacsv.WriteOptions(quoting_style="needed"))
        payload = buffer.getvalue()
        if current == "csv_gz":
            (root / f"{stem}.csv.gz").write_bytes(gzip.compress(payload, mtime=0))
        else:
            (root / f"{stem}.csv").write_bytes(payload)


def _cells_table(
    profile: Profile,
    options: Options,
    nuclei: tuple[tuple[int, Rect], ...],
    counts: NDArray[np.int32],
    types: tuple[str, ...],
) -> pa.Table:
    def by_type(*names: str) -> NDArray[np.int64]:
        columns = [index for index, name in enumerate(types) if name in names]
        return counts[:, columns].sum(axis=1).astype(np.int64)

    centroid_x: list[float] = []
    centroid_y: list[float] = []
    cell_area: list[float] = []
    nucleus_area: list[float] = []
    for cell, rect in enumerate(CELL_RECTS):
        centroid_x.append(((rect.left + rect.right) / 2 + 0.5) * PIXEL_SIZE)
        centroid_y.append(((rect.top + rect.bottom) / 2 + 0.5) * PIXEL_SIZE)
        cell_area.append(rect.pixels * PIXEL_SIZE**2)
        owned = [nucleus.pixels for owner, nucleus in nuclei if owner == cell]
        nucleus_area.append(sum(owned) * PIXEL_SIZE**2 if owned else float("nan"))

    gene = by_type("Gene Expression")
    probe = by_type("Negative Control Probe")
    codeword = by_type("Negative Control Codeword")
    unassigned = by_type("Unassigned Codeword", "Blank Codeword")
    genomic = by_type("Genomic Control")
    deprecated = by_type("Deprecated Codeword")
    total = gene + probe + codeword + unassigned + genomic + deprecated

    columns: dict[str, pa.Array] = {
        "cell_id": _id_array(profile, _ids(profile)),
        "x_centroid": pa.array(centroid_x, pa.float64()),
        "y_centroid": pa.array(centroid_y, pa.float64()),
        "transcript_counts": pa.array(gene),
        "control_probe_counts": pa.array(probe),
    }
    if profile.generation in ("3.0", "4.0"):
        columns["genomic_control_counts"] = pa.array(genomic)
    columns["control_codeword_counts"] = pa.array(codeword)
    if profile.generation != "1.0":
        columns["unassigned_codeword_counts"] = pa.array(unassigned)
    if profile.generation in ("1.5", "2.0", "3.0", "4.0"):
        columns["deprecated_codeword_counts"] = pa.array(deprecated)
    columns["total_counts"] = pa.array(total)
    columns["cell_area"] = pa.array(cell_area, pa.float64())
    columns["nucleus_area"] = pa.array(nucleus_area, pa.float64(), from_pandas=False)
    if profile.generation in ("3.0", "4.0"):
        columns["nucleus_count"] = pa.array(
            [sum(owner == cell for owner, _ in nuclei) for cell in range(len(CELL_RECTS))], pa.int64()
        )
        methods = [SEGMENTATION_METHODS[IMPORTED_METHOD if options.import_kind else m] for m in CELL_METHODS]
        columns["segmentation_method"] = pa.array(methods, pa.string())
    return pa.table(columns)


def _boundaries(
    profile: Profile, polygons: list[tuple[str | int, int, list[tuple[float, float]]]], *, label_id: bool
) -> pa.Table:
    ids: list[str | int] = []
    xs: list[float] = []
    ys: list[float] = []
    labels: list[int] = []
    for cell_id, label, ring in polygons:
        for x, y in ring:
            ids.append(cell_id)
            xs.append(x)
            ys.append(y)
            labels.append(label)
    columns = {
        "cell_id": _id_array(profile, ids),
        "vertex_x": pa.array(xs, pa.float32()),
        "vertex_y": pa.array(ys, pa.float32()),
    }
    if label_id:
        columns["label_id"] = pa.array(labels, pa.int64())
    return pa.table(columns)


def _transcript_table(
    profile: Profile, transcripts: Transcripts, features: tuple[tuple[str, str, str], ...]
) -> pa.Table:
    names = [features[index][1] for index in transcripts.feature_index]
    if profile.family == "integer_ids":
        cell_ids = pa.array([INTEGER_IDS[c] if c >= 0 else -1 for c in transcripts.cell_index], pa.int32())
        feature_names = pa.array([name.encode() for name in names], pa.binary())
    else:
        values = [HEX_IDS[c] if c >= 0 else "UNASSIGNED" for c in transcripts.cell_index]
        cell_ids = _id_array(profile, values)
        feature_names = pa.array(names, pa.string())
    columns: dict[str, pa.Array] = {
        "transcript_id": pa.array(transcripts.transcript_id),
        "cell_id": cell_ids,
        "overlaps_nucleus": pa.array(transcripts.overlaps_nucleus),
        "feature_name": feature_names,
        "x_location": pa.array(transcripts.x),
        "y_location": pa.array(transcripts.y),
        "z_location": pa.array(transcripts.z),
        "qv": pa.array(transcripts.qv),
    }
    if profile.generation != "1.0":
        columns["fov_name"] = pa.array(["A1" if x < 16 * PIXEL_SIZE else "B1" for x in transcripts.x], pa.string())
        columns["nucleus_distance"] = pa.array(np.zeros(len(names), dtype=np.float32))
    if profile.generation in ("2.0", "3.0", "4.0"):
        columns["codeword_index"] = pa.array(transcripts.feature_index.astype(np.int32))
    if profile.generation in ("3.0", "4.0"):
        categories = {"Gene Expression": "predesigned_gene", "Negative Control Probe": "negative_control_probe"}
        category = [categories.get(features[index][2], "unassigned_codeword") for index in transcripts.feature_index]
        columns["codeword_category"] = pa.array(category, pa.string())
        columns["is_gene"] = pa.array([value == "predesigned_gene" for value in category], pa.bool_())
    return pa.table(columns)


def _write_h5(
    path: Path,
    profile: Profile,
    features: tuple[tuple[str, str, str], ...],
    counts: NDArray[np.int32],
    barcodes: tuple[str, ...],
) -> None:
    """Write a 10x feature-barcode HDF5 matrix (features x cells, CSC over cells)."""
    matrix = counts.T
    data: list[int] = []
    indices: list[int] = []
    indptr = [0]
    for column in range(matrix.shape[1]):
        rows = np.flatnonzero(matrix[:, column])
        data.extend(matrix[rows, column].tolist())
        indices.extend(rows.tolist())
        indptr.append(len(data))
    with h5py.File(path, "w") as handle:
        handle.attrs["filetype"] = "matrix"
        handle.attrs["library_ids"] = np.array([b"0000000_0000000"])
        handle.attrs["original_gem_groups"] = np.array([1])
        handle.attrs["version"] = 2
        if profile.protein:
            handle.attrs["protein_scaling_factor"] = np.int64(PROTEIN_FACTOR)
            handle.attrs["protein_metric"] = "MEAN_PER_CELL_STAIN"
        group = handle.create_group("matrix")
        group["barcodes"] = np.array([barcode.encode() for barcode in barcodes], dtype="S10")
        group["data"] = np.asarray(data, dtype=np.int32)
        group["indices"] = np.asarray(indices, dtype=np.int64)
        group["indptr"] = np.asarray(indptr, dtype=np.int64)
        group["shape"] = np.asarray(matrix.shape, dtype=np.int32)
        feature_group = group.create_group("features")
        feature_group["_all_tag_keys"] = np.array([b"genome"])
        feature_group["feature_type"] = np.array([feature[2].encode() for feature in features])
        feature_group["genome"] = np.array([b"Unknown"] * len(features))
        feature_group["id"] = np.array([feature[0].encode() for feature in features])
        feature_group["name"] = np.array([feature[1].encode() for feature in features])


def _mtx_members(
    features: tuple[tuple[str, str, str], ...], counts: NDArray[np.int32], barcodes: tuple[str, ...]
) -> dict[str, bytes]:
    matrix = counts.T
    rows, columns = np.nonzero(matrix)
    order = np.lexsort((rows, columns))
    lines = [
        "%%MatrixMarket matrix coordinate integer general",
        "%metadata_json: {}",
        f"{matrix.shape[0]} {matrix.shape[1]} {len(rows)}",
        *(f"{rows[i] + 1} {columns[i] + 1} {matrix[rows[i], columns[i]]}" for i in order),
    ]
    features_text = "".join(f"{feature_id}\t{name}\t{kind}\n" for feature_id, name, kind in features)
    return {
        "barcodes.tsv.gz": gzip.compress("".join(f"{b}\n" for b in barcodes).encode(), mtime=0),
        "features.tsv.gz": gzip.compress(features_text.encode(), mtime=0),
        "matrix.mtx.gz": gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0),
    }


def _ome_xml(
    channels: tuple[str, ...], files: tuple[str, ...], uuids: tuple[str, ...], own: int, *, dtype: str = "uint16"
) -> str:
    channel_xml = "".join(
        f'<Channel ID="Channel:{i}" Name="{name}" SamplesPerPixel="1"/>' for i, name in enumerate(channels)
    )
    tiff_data = "".join(
        f'<TiffData FirstC="{i}" FirstT="0" FirstZ="0" IFD="0" PlaneCount="1">'
        f'<UUID FileName="{file}">urn:uuid:{identifier}</UUID></TiffData>'
        for i, (file, identifier) in enumerate(zip(files, uuids, strict=True))
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" UUID="urn:uuid:{uuids[own]}">'
        f'<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="{dtype}" SizeX="{SHAPE[1]}" '
        f'SizeY="{SHAPE[0]}" SizeZ="1" SizeC="{len(channels)}" SizeT="1" PhysicalSizeX="{PIXEL_SIZE}" '
        f'PhysicalSizeXUnit="&#181;m" PhysicalSizeY="{PIXEL_SIZE}" PhysicalSizeYUnit="&#181;m">'
        f"{channel_xml}{tiff_data}</Pixels></Image></OME>"
    )


def write_focus_set(
    directory: Path, names: tuple[str, ...], channels: tuple[str, ...], pixels: NDArray[np.uint16]
) -> None:
    """Write one single-plane OME-TIFF per channel, each listing the complete UUID-linked set."""
    directory.mkdir(parents=True, exist_ok=True)
    uuids = tuple(str(uuid.uuid5(uuid.NAMESPACE_URL, f"synthetic/{name}")) for name in names)
    for index, name in enumerate(names):
        xml = _ome_xml(channels, names, uuids, index)
        tifffile.imwrite(directory / name, pixels[index], description=xml, metadata=None, tile=(16, 16))


def _focus_pixels(channels: int) -> NDArray[np.uint16]:
    base = np.arange(SHAPE[0] * SHAPE[1], dtype=np.uint16).reshape(SHAPE)
    return np.stack([base + 1000 * index for index in range(channels)]).astype(np.uint16)


def _slug(name: str) -> str:
    return name.lower().replace("/", "_").replace(" ", "_")


def _focus_names(profile: Profile) -> tuple[str, ...]:
    if profile.images == "single":
        return ("morphology_focus.ome.tif",)
    if profile.images == "numbered":
        return tuple(f"morphology_focus_{index:04d}.ome.tif" for index in range(len(profile.channels)))
    return tuple(f"ch{index:04d}_{_slug(name)}.ome.tif" for index, name in enumerate(profile.channels))


def _image_entries(profile: Profile) -> dict[str, str]:
    """Return the experiment.xenium images block; stripped deposits keep it without the files."""
    entries = {"morphology_filepath": "morphology.ome.tif"}
    if profile.images == "single":
        return entries | {
            "morphology_mip_filepath": "morphology_mip.ome.tif",
            "morphology_focus_filepath": "morphology_focus.ome.tif",
        }
    return entries | {"morphology_focus_filepath": f"morphology_focus/{_focus_names(profile)[0]}"}


def _write_images(root: Path, profile: Profile, options: Options, focus: NDArray[np.uint16]) -> None:
    names = _focus_names(profile)
    if profile.images == "single":
        write_focus_set(root, names, profile.channels, focus)
        if options.mip:
            write_focus_set(root, ("morphology_mip.ome.tif",), profile.channels, focus + np.uint16(500))
    else:
        write_focus_set(root / "morphology_focus", names, profile.channels, focus)


def _write_aligned(root: Path, kinds: tuple[Literal["he", "if"], ...]) -> None:
    if "he" in kinds:
        rgb = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        tifffile.imwrite(
            root / "synthetic_he_image.ome.tif", rgb, photometric="rgb", ome=True, metadata={"axes": "YXS"}
        )
        (root / "synthetic_he_imagealignment.csv").write_text("0,-0.5,10.0\n0.5,0,2.0\n0,0,1\n")
    if "if" in kinds:
        data = np.arange(2 * 12 * 16, dtype=np.uint16).reshape(2, 12, 16)
        tifffile.imwrite(
            root / "synthetic_if_image.ome.tif",
            data,
            ome=True,
            metadata={"axes": "CYX", "Channel": {"Name": ["CD20", "DAPI"]}},
        )


def _cell_summary(profile: Profile, fixture: XeniumFixture) -> tuple[list[str], NDArray[np.float64]]:
    """Return the ``cell_summary`` column names and rows from the mask pixels."""
    columns = [
        "cell_centroid_x",
        "cell_centroid_y",
        "cell_area",
        "nucleus_centroid_x",
        "nucleus_centroid_y",
        "nucleus_area",
        "z_level",
    ]
    summary: list[list[float]] = []
    for cell, rect in enumerate(CELL_RECTS):
        owned = [nucleus for owner, nucleus in fixture.nuclei if owner == cell]
        pixels = sum(nucleus.pixels for nucleus in owned)
        if owned:
            weights = [nucleus.pixels for nucleus in owned]
            nucleus_x = float(np.average([((n.left + n.right) / 2 + 0.5) * PIXEL_SIZE for n in owned], weights=weights))
            nucleus_y = float(np.average([((n.top + n.bottom) / 2 + 0.5) * PIXEL_SIZE for n in owned], weights=weights))
        else:
            nucleus_x = nucleus_y = float("nan")
        summary.append(
            [
                ((rect.left + rect.right) / 2 + 0.5) * PIXEL_SIZE,
                ((rect.top + rect.bottom) / 2 + 0.5) * PIXEL_SIZE,
                rect.pixels * PIXEL_SIZE**2,
                nucleus_x,
                nucleus_y,
                pixels * PIXEL_SIZE**2 if owned else float("nan"),
                float(cell + 3),
            ]
        )
    if profile.family == "polygon_sets":
        columns.append("nucleus_count")
        for cell, row in enumerate(summary):
            row.append(float(sum(owner == cell for owner, _ in fixture.nuclei)))
    return columns, np.asarray(summary, dtype=np.float64)


def _write_polygon_sets(group: zarr.Group, options: Options, fixture: XeniumFixture) -> dict[str, JSON]:
    """Write the XOA 2.0+ polygon sets and return their root attributes."""
    n = len(CELL_RECTS)
    names = ["cell"] if options.import_kind == "transcripts" else ["nucleus", "cell"]
    sets = {"nucleus": [owner for owner, _ in fixture.nuclei], "cell": list(range(n))}
    cell_methods = [IMPORTED_METHOD] * n if options.import_kind else list(CELL_METHODS)
    methods = {"nucleus": [3] * len(fixture.nuclei), "cell": cell_methods}
    for index, name in enumerate(names):
        polygon_set = group.require_group(f"polygon_sets/{index}")
        size = len(sets[name])
        for key, values, dtype in (
            ("cell_index", sets[name], np.uint32),
            ("method", methods[name], np.uint32),
            ("num_vertices", [5] * size, np.int32),
        ):
            polygon_set.create_array(
                key, data=np.asarray(values, dtype=dtype), chunks=(max(size, 1),), compressors=None
            )
    return {
        "major_version": 6,
        "minor_version": 2,
        "polygon_set_names": names,
        "polygon_set_display_names": [f"{name.title()} boundaries" for name in names],
        "segmentation_methods": list(SEGMENTATION_METHODS),
    }


def zip_tree(tree: Path, archive_path: Path) -> None:
    """Pack a zarr v2 tree as a stored zip in sorted member order and remove the tree."""
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(tree).as_posix())
    shutil.rmtree(tree)


def _write_cells_store(root: Path, profile: Profile, options: Options, fixture: XeniumFixture) -> None:
    """Write cells.zarr.zip as a stored zip of a zarr v2 tree, as the vendor does."""
    tree = root / ".cells_tree"
    group = zarr.open_group(store=LocalStore(tree), mode="w", zarr_format=2)
    n = len(CELL_RECTS)
    attributes: dict[str, JSON] = {"name": "CellSegmentationDataset", "number_cells": n, "spatial_units": "microns"}
    if profile.family == "integer_ids":
        attributes |= {"major_version": 4, "minor_version": 0, "cell_boundary_polygon_set_idx": 0}
        group.create_array("cell_id", data=np.asarray(INTEGER_IDS, dtype=np.uint32), chunks=(n,), compressors=None)
    else:
        group.create_array("cell_id", data=np.asarray(HEX_PAIRS, dtype=np.uint32), chunks=(n, 2), compressors=None)
    if profile.family == "hex_ids":
        attributes |= {"major_version": 5, "minor_version": 0}
        group.create_array(
            "seg_mask_value", data=np.asarray(fixture.cell_labels, dtype=np.uint32), chunks=(n,), compressors=None
        )

    columns, summary = _cell_summary(profile, fixture)
    cell_summary = group.create_array("cell_summary", data=summary, chunks=(n, 1), compressors=None)
    cell_summary.attrs.update({"column_names": columns, "column_descriptions": columns})

    if profile.family == "polygon_sets":
        attributes |= _write_polygon_sets(group, options, fixture)
    else:
        attributes["polygon_set_names"] = ["nucleus", "cell"]
    group.attrs.update(attributes)

    if options.import_kind != "transcripts":
        masks = group.require_group("masks")
        masks.create_array("0", data=fixture.nucleus_mask, chunks=MASK_CHUNKS, compressors=None)
        masks.create_array("1", data=fixture.cell_mask, chunks=MASK_CHUNKS, compressors=None)
        transform = np.diag([1 / PIXEL_SIZE, 1 / PIXEL_SIZE, 1.0, 1.0]).astype(np.float32)
        masks.create_array("homogeneous_transform", data=transform, chunks=(4, 4), compressors=None)

    zip_tree(tree, root / "cells.zarr.zip")


def _write_metadata(root: Path, profile: Profile) -> None:
    (root / "metrics_summary.csv").write_text(
        "run_name,region_name,num_cells_detected,fraction_transcripts_assigned,median_genes_per_cell,"
        "number_of_cells_non_simple_polygons\nsynthetic,region,4,0.875,2.5,\n"
    )
    panel = {
        "metadata": {"sha256_checksum": "0" * 64},
        "payload": {
            "panel": {
                "identity": {"design_id": "synthetic_v1", "name": "Synthetic panel", "version": "1.0.0"},
                "num_gene_targets": len(GENES),
                "species": "Human",
                "tissue": "Synthetic",
            },
            "spec_version": "0.1.0",
            "targets": [
                {"type": {"data": {"id": gene_id, "name": name}, "descriptor": "gene"}} for gene_id, name in GENES
            ],
        },
    }
    (root / "gene_panel.json").write_text(json.dumps(panel))
    if profile.protein:
        protein_panel = {
            "payload": {
                "panel": {
                    "identity": {"design_id": "protein-panel-synthetic", "name": "protein-panel-synthetic"},
                    "num_targets": len(PROTEINS),
                    "panel_type": "PANEL_TYPE_PROTEIN",
                    "spec_version": "1.0.0",
                    "species": "unknown",
                    "targets": [
                        {
                            "public_id": public_id,
                            "short_name": name,
                            "long_name": f"{name} protein",
                            "clone": "n/a",
                            "codeword_index": 16 + index,
                        }
                        for index, (public_id, name) in enumerate(PROTEINS)
                    ],
                }
            }
        }
        (root / "protein_panel.json").write_text(json.dumps(protein_panel))


def _write_analysis(root: Path, kind: Literal["archive", "directory"], barcodes: tuple[str, ...]) -> None:
    """Write clustering and projection CSVs; the last cell is absent, as filtered cells are."""
    kept = barcodes[:-1]
    members = {
        "analysis/clustering/gene_expression_graphclust/clusters.csv": "Barcode,Cluster\n"
        + "".join(f"{barcode},{index % 2 + 1}\n" for index, barcode in enumerate(kept)),
        "analysis/pca/gene_expression_3_components/projection.csv": "Barcode,PC-1,PC-2,PC-3\n"
        + "".join(f"{barcode},{index}.5,-{index}.25,0.125\n" for index, barcode in enumerate(kept)),
        "analysis/umap/gene_expression_2_components/projection.csv": "Barcode,UMAP-1,UMAP-2\n"
        + "".join(f"{barcode},{index}.75,{index}.0\n" for index, barcode in enumerate(kept)),
        "analysis/diffexp/gene_expression_graphclust/differential_expression.csv": "Feature ID,Feature Name\n",
    }
    if kind == "directory":
        for name, text in members.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text)
        return
    with tarfile.open(root / "analysis.tar.gz", "w:gz") as archive:
        for name, text in members.items():
            payload = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _experiment(profile: Profile, options: Options, images: dict[str, str]) -> dict[str, object]:
    experiment: dict[str, object] = {}
    if profile.version is not None:
        major, minor, patch = profile.version
        experiment |= {"major_version": major, "minor_version": minor}
        if patch is not None:
            experiment["patch_version"] = patch
    experiment |= {
        "run_name": "synthetic run",
        "region_name": "region",
        "panel_name": "Synthetic panel",
        "num_cells": len(CELL_RECTS),
        "pixel_size": PIXEL_SIZE,
        "instrument_sw_version": "R&D",
        "analysis_sw_version": profile.software,
        "z_step_size": Z_STEP,
        "images": images,
        "xenium_explorer_files": {"cells_zarr_filepath": "cells.zarr.zip"},
    }
    if options.ranger is not None:
        experiment["xenium_ranger"] = {
            "run_id": "synthetic_XRrun",
            "version": "xenium-3.1.0.5",
            "command_line": f"xeniumranger {options.ranger} --id=synthetic --xenium-bundle=/home/user/private/outs",
            "original_analysis_uuid": "fd0b49ea-0000-4000-8000-000000000000",
        }
    return experiment


def _write_boundaries(root: Path, profile: Profile, options: Options, fixture: XeniumFixture) -> None:
    cell_ids, nuclei = fixture.cell_ids, fixture.nuclei
    cell_polygons = [
        (cell_ids[cell], cell + 1, rect.ring(bowtie=cell in options.bowtie_cells))
        for cell, rect in enumerate(CELL_RECTS)
    ]
    _write_table(
        _boundaries(profile, cell_polygons, label_id=profile.family == "polygon_sets"),
        root,
        "cell_boundaries",
        kind=options.tables,
        twins=options.csv_twins,
    )
    if options.import_kind == "transcripts":
        return

    nucleus_polygons = []
    for index, (owner, rect) in enumerate(nuclei):
        ring = rect.ring(bowtie=index in options.bowtie_nuclei)
        if index in options.short_nuclei:
            ring = [ring[0], ring[1], ring[0]]
        nucleus_polygons.append((cell_ids[owner], fixture.nucleus_labels[index], ring))
    label_column = profile.family == "polygon_sets" and options.nucleus_label_id
    _write_table(
        _boundaries(profile, nucleus_polygons, label_id=label_column),
        root,
        "nucleus_boundaries",
        kind=options.tables,
        twins=options.csv_twins,
    )


def _write_transcripts(
    root: Path, profile: Profile, options: Options, features: tuple[tuple[str, str, str], ...], transcripts: Transcripts
) -> None:
    if options.transcripts == "parquet":
        table = _transcript_table(profile, transcripts, features)
        pq.write_table(table, root / "transcripts.parquet", row_group_size=options.transcript_row_group)
    elif options.transcripts is not None:
        _write_table(
            _transcript_table(profile, transcripts, features),
            root,
            "transcripts",
            kind=options.transcripts,
            twins=False,
        )


def _write_counts(
    root: Path, profile: Profile, options: Options, features: tuple[tuple[str, str, str], ...], fixture: XeniumFixture
) -> None:
    for kind in options.counts:
        if kind == "h5":
            _write_h5(root / "cell_feature_matrix.h5", profile, features, fixture.counts, fixture.barcodes)
            continue
        members = _mtx_members(features, fixture.counts, fixture.barcodes)
        if kind == "mtx":
            directory = root / "cell_feature_matrix"
            directory.mkdir(exist_ok=True)
            for name, payload in members.items():
                (directory / name).write_bytes(payload)
        else:
            with tarfile.open(root / "cell_feature_matrix.tar.gz", "w:gz") as archive:
                for name, payload in members.items():
                    info = tarfile.TarInfo(f"cell_feature_matrix/{name}")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))


def write_xenium_bundle(root: Path, profile: Profile, options: Options | None = None) -> XeniumFixture:
    """Write one synthetic bundle and return its independently known truth."""
    options = options or Options()
    root.mkdir(parents=True, exist_ok=True)
    features = _features(profile)
    nuclei = _nuclei(profile, options)
    nucleus_labels = (
        tuple(range(1, len(nuclei) + 1))
        if profile.family == "polygon_sets"
        else tuple(owner + 1 if profile.family == "hex_ids" else INTEGER_IDS[owner] for owner, _ in nuclei)
    )
    cell_ids = _ids(profile)
    cell_labels = tuple(range(1, len(CELL_RECTS) + 1))
    cell_mask, nucleus_mask = _masks(nuclei, nucleus_labels)
    transcripts = _transcripts(nuclei, sum(kind != "Protein Expression" for *_, kind in features))
    counts = _counts(transcripts, features)
    focus = _focus_pixels(len(profile.channels))

    if options.images:
        _write_images(root, profile, options, focus)
    experiment = _experiment(profile, options, _image_entries(profile))
    (root / "experiment.xenium").write_text(json.dumps(experiment, indent=4))

    fixture = XeniumFixture(
        root=root,
        profile=profile,
        options=options,
        cell_ids=cell_ids,
        barcodes=tuple(str(cell_id) for cell_id in cell_ids),
        cell_labels=cell_labels,
        nuclei=nuclei,
        nucleus_labels=nucleus_labels,
        features=tuple(name for _, name, _ in features),
        feature_ids=tuple(feature_id for feature_id, _, _ in features),
        feature_types=tuple(kind for *_, kind in features),
        counts=counts,
        transcripts=transcripts,
        cell_mask=cell_mask,
        nucleus_mask=nucleus_mask,
        focus=focus,
        focus_entry=_image_entries(profile)["morphology_focus_filepath"],
        channels=profile.channels,
        experiment=experiment,
    )

    _write_table(
        _cells_table(profile, options, nuclei, counts, fixture.feature_types),
        root,
        "cells",
        kind=options.tables,
        twins=options.csv_twins,
    )
    _write_boundaries(root, profile, options, fixture)
    _write_transcripts(root, profile, options, features, transcripts)
    _write_counts(root, profile, options, features, fixture)

    if options.cells_store:
        _write_cells_store(root, profile, options, fixture)
    if options.metadata:
        _write_metadata(root, profile)
    if options.analysis is not None:
        _write_analysis(root, options.analysis, fixture.barcodes)
    _write_aligned(root, options.aligned)
    return fixture
