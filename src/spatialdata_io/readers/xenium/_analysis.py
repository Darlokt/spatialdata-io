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
import re
import tarfile
from contextlib import closing
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
    from collections.abc import Generator
    from pathlib import Path
    from typing import IO

    from numpy.typing import NDArray

    from spatialdata_io.readers.xenium._layout import AnalysisSource

BARCODE = "Barcode"
CLUSTER = "Cluster"
_MEMBER = re.compile(
    r"(?:analysis/)?(?P<kind>clustering|pca|umap)/(?P<method>[A-Za-z0-9_.-]+)/(?P<file>clusters|projection)\.csv"
)
_FILES = {"clustering": "clusters", "pca": "projection", "umap": "projection"}
_PREFIXES = {"pca": "PC", "umap": "UMAP"}
_MIN_PROJECTION_COLUMNS = 2


@dataclass(frozen=True, slots=True)
class Analysis:
    """Per-cell clusterings and projections in the order of the given barcodes."""

    clusters: dict[str, pd.Categorical]
    projections: dict[str, NDArray[np.float64]]


def _identity(name: str, seen: set[tuple[str, str]], path: Path) -> tuple[str, str] | None:
    """Reject ambiguous semantic artifacts before reading any selected payload."""
    match = _MEMBER.fullmatch(name)
    if match is None or match["file"] != _FILES[match["kind"]]:
        return None
    identity = match["kind"], match["method"]
    if identity in seen:
        raise schema_error(
            path, "analysis", found=f"repeated analysis identity {identity!r}", expected="one CSV per kind and method"
        )
    seen.add(identity)
    return identity


def _members(source: AnalysisSource) -> Generator[tuple[str, str, IO[bytes], Path], None, None]:
    """Own selected CSV streams after rejecting repeated semantic identities."""
    seen: set[tuple[str, str]] = set()
    if isinstance(source, AnalysisDirectory):
        files: list[tuple[str, str, Path]] = []
        path: Path
        for path in sorted(source.directory.rglob("*.csv")):
            identity = _identity(path.relative_to(source.directory).as_posix(), seen, path)
            if identity is not None:
                files.append((*identity, path))
        for kind, method, path in files:
            with path.open("rb") as handle:
                yield kind, method, handle, path
        return
    with tarfile.open(source.file, "r:gz") as archive:
        members: list[tuple[str, str, tarfile.TarInfo]] = []
        for member in sorted((member for member in archive if member.isfile()), key=lambda member: member.name):
            identity = _identity(str(PurePosixPath(member.name)), seen, source.file)
            if identity is not None:
                members.append((*identity, member))
        for kind, method, member in members:
            member_handle = archive.extractfile(member)
            if member_handle is None:  # pragma: no cover - isfile() members always have a payload
                raise schema_error(source.file, member.name, found="no payload", expected="a regular analysis CSV")
            with member_handle:
                yield kind, method, member_handle, source.file


def _header(handle: IO[bytes], path: Path) -> list[str]:
    try:
        line = handle.readline()
        header = next(csv.reader([line.decode("utf-8-sig")], strict=True), [])
    except (UnicodeDecodeError, csv.Error) as error:
        raise schema_error(path, "analysis", found="a malformed header", expected="one UTF-8 CSV header") from error
    # Arrow reads the header itself; no encoded body buffer is retained.
    handle.seek(0)
    return header


def _table(handle: IO[bytes], header: list[str], value: pa.DataType, path: Path) -> pa.Table:
    """Parse an owned CSV stream directly with text barcodes and typed values."""
    try:
        return pacsv.read_csv(
            handle,
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
        nonpositive clusters, unknown or repeated barcodes, or repeated semantic
        artifacts.
    """
    clusters: dict[str, pd.Categorical] = {}
    projections: dict[str, NDArray[np.float64]] = {}
    with closing(_members(source)) as members:
        for kind, method, handle, path in members:
            header = _header(handle, path)
            if kind == "clustering":
                if header != [BARCODE, CLUSTER]:
                    raise schema_error(path, "analysis", found=f"header {header!r}", expected="Barcode,Cluster")
                clusters[method] = _clusters(_table(handle, header, pa.int64(), path), barcodes, path)
                continue

            prefix = _PREFIXES[kind]
            expected = [BARCODE, *(f"{prefix}-{index}" for index in range(1, len(header)))]
            if len(header) < _MIN_PROJECTION_COLUMNS or header != expected:
                raise schema_error(path, "analysis", found=f"header {header!r}", expected=f"Barcode,{prefix}-1,...")
            projections[f"{kind}_{method}"] = _projection(
                _table(handle, header, pa.float64(), path), header, barcodes, path
            )
    return Analysis(clusters, projections)
