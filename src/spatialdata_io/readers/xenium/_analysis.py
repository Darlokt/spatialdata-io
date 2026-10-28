"""Read opt-in Xenium secondary analysis aligned to the cells by exact barcode.

XOA writes an extracted ``analysis/`` directory (1.0.x) or ``analysis.tar.gz``
(1.9+) with one schema from XOA 1.0.2 to 4.0:

- ``clustering/<method>/clusters.csv``: ``Barcode,Cluster`` with positive
  integer clusters, becoming the categorical observation column ``<method>``
  whose text categories (``"1"``, ``"2"``, ...) follow numeric order, as
  scanpy clusterings do;
- ``pca/<method>/projection.csv`` and ``umap/<method>/projection.csv``:
  ``Barcode,PC-1..`` or ``Barcode,UMAP-1..``, becoming float64
  ``obsm["pca_<method>"]`` and ``obsm["umap_<method>"]``.

10x omits cells it filtered before clustering; they become missing values.
Differential expression, PCA loadings and protein z-score summaries are not
per-cell data and stay unread.
"""

from __future__ import annotations

import csv
import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from spatialdata_io.readers.xenium._layout import AnalysisDirectory
from spatialdata_io.readers.xenium._schemas import schema_error, text_series

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from spatialdata_io.readers.xenium._layout import AnalysisSource

MAX_MEMBER_BYTES = 2 * 1024**3
BARCODE = "Barcode"
CLUSTER = "Cluster"
_MEMBER = re.compile(
    r"(?:analysis/)?(?P<kind>clustering|pca|umap)/(?P<method>[A-Za-z0-9_.-]+)/(?P<file>clusters|projection)\.csv"
)
_FILES = {"clustering": "clusters", "pca": "projection", "umap": "projection"}
_PREFIXES = {"pca": "PC", "umap": "UMAP"}


@dataclass(frozen=True, slots=True)
class Analysis:
    """Per-cell clusterings and projections in the order of the given barcodes."""

    clusters: dict[str, pd.Categorical]
    projections: dict[str, NDArray[np.float64]]


def _members(source: AnalysisSource) -> Iterator[tuple[str, str, bytes, Path]]:
    """Yield (kind, method, payload, path) of the per-cell CSVs in sorted order."""
    if isinstance(source, AnalysisDirectory):
        for path in sorted(source.directory.rglob("*.csv")):
            match = _MEMBER.fullmatch(path.relative_to(source.directory).as_posix())
            if match and match["file"] == _FILES[match["kind"]]:
                yield match["kind"], match["method"], path.read_bytes(), path
        return
    with tarfile.open(source.file, "r:gz") as archive:
        members = sorted((member for member in archive if member.isfile()), key=lambda member: member.name)
        for member in members:
            match = _MEMBER.fullmatch(str(PurePosixPath(member.name)))
            if not match or match["file"] != _FILES[match["kind"]]:
                continue
            if member.size > MAX_MEMBER_BYTES:
                raise schema_error(
                    source.file, member.name, found=f"{member.size} bytes", expected="a member within 2 GiB"
                )
            handle = archive.extractfile(member)
            if handle is None:  # pragma: no cover - isfile() members always have a payload
                continue
            yield match["kind"], match["method"], handle.read(), source.file


def _header(payload: bytes, path: Path) -> list[str]:
    try:
        line = payload.split(b"\n", 1)[0].decode("utf-8-sig")
        return next(csv.reader([line], strict=True), [])
    except (UnicodeDecodeError, csv.Error) as error:
        raise schema_error(path, "analysis", found="a malformed header", expected="one UTF-8 CSV header") from error


def _table(payload: bytes, header: list[str], value: pa.DataType, path: Path) -> pa.Table:
    """Parse one analysis CSV with text barcodes and typed values."""
    try:
        return pacsv.read_csv(
            io.BytesIO(payload),
            read_options=pacsv.ReadOptions(use_threads=False),
            convert_options=pacsv.ConvertOptions(
                column_types={BARCODE: pa.string(), **dict.fromkeys(header[1:], value)},
                strings_can_be_null=False,
            ),
        )
    except pa.ArrowInvalid as error:
        raise schema_error(path, "analysis", found=f"malformed CSV: {error}", expected=f"columns {header!r}") from error


def _rows(table: pa.Table, barcodes: pd.Index, path: Path) -> NDArray[np.intp]:
    """Return the cell row of every record; barcodes must be unique and known."""
    rows = barcodes.get_indexer(pd.Index(text_series(table[BARCODE])))
    if (rows < 0).any() or not pd.Index(rows).is_unique:
        raise schema_error(
            path, "analysis", found="unknown or repeated barcodes", expected="unique barcodes of known cells"
        )
    return rows


def _clusters(table: pa.Table, barcodes: pd.Index, path: Path) -> pd.Categorical:
    values = np.asarray(table[CLUSTER].to_numpy(), dtype=np.int64)
    if (values <= 0).any():
        raise schema_error(path, "analysis", found="nonpositive clusters", expected="positive integer clusters")
    categories = np.unique(values)
    codes = np.full(len(barcodes), -1, dtype=np.int64)
    codes[_rows(table, barcodes, path)] = np.searchsorted(categories, values)
    return pd.Categorical.from_codes(codes, categories=pd.Index(categories.astype(np.str_), dtype="str"))


def _projection(table: pa.Table, header: list[str], barcodes: pd.Index, path: Path) -> NDArray[np.float64]:
    values = np.column_stack([np.asarray(table[name].to_numpy(), dtype=np.float64) for name in header[1:]])
    if not np.isfinite(values).all():
        raise schema_error(path, "analysis", found="nonfinite projections", expected="finite projections")
    matrix = np.full((len(barcodes), len(header) - 1), np.nan)
    matrix[_rows(table, barcodes, path)] = values
    return matrix


def read_analysis(source: AnalysisSource, barcodes: pd.Index) -> Analysis:
    """Align every clustering and projection to ``barcodes``; absent cells are missing.

    Raises
    ------
    ReaderFormatError
        If a CSV has an unexpected header, malformed or nonfinite values,
        nonpositive clusters, or unknown or repeated barcodes.
    """
    clusters: dict[str, pd.Categorical] = {}
    projections: dict[str, NDArray[np.float64]] = {}
    for kind, method, payload, path in _members(source):
        header = _header(payload, path)
        if kind == "clustering":
            if header != [BARCODE, CLUSTER]:
                raise schema_error(path, "analysis", found=f"header {header!r}", expected="Barcode,Cluster")
            clusters[method] = _clusters(_table(payload, header, pa.int64(), path), barcodes, path)
            continue

        prefix = _PREFIXES[kind]
        expected = [BARCODE, *(f"{prefix}-{index}" for index in range(1, len(header)))]
        if len(header) < 2 or header != expected:
            raise schema_error(path, "analysis", found=f"header {header!r}", expected=f"Barcode,{prefix}-1,...")
        projections[f"{kind}_{method}"] = _projection(
            _table(payload, header, pa.float64(), path), header, barcodes, path
        )
    return Analysis(clusters, projections)
