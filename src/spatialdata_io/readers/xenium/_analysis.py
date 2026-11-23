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
Each kind and method must name exactly one CSV. Methods are compared ignoring
letter case, as SpatialData compares annotation names. Differential expression,
PCA loadings and protein z-score summaries are not per-cell data and stay
unread.
"""

from __future__ import annotations

import re
import tarfile
from contextlib import closing
from dataclasses import dataclass
from io import RawIOBase
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, override

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from spatialdata_io.readers._utils.csv import csv_errors, csv_header
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.xenium._layout import AnalysisDirectory
from spatialdata_io.readers.xenium._schemas import BARCODE, CLUSTER, arrow_series, validate_analysis_header

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from typing import IO

    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers.xenium._layout import AnalysisSource

_MEMBER = re.compile(
    r"(?:analysis/)?(?P<kind>clustering|pca|umap)/(?P<method>[A-Za-z0-9_.-]+)/(?P<file>clusters|projection)\.csv"
)
_FILES = {"clustering": "clusters", "pca": "projection", "umap": "projection"}


@dataclass(frozen=True, slots=True)
class Analysis:
    """Per-cell clusterings and projections in the order of the given barcodes."""

    clusters: dict[str, pd.Categorical]
    projections: dict[str, NDArray[np.float64]]


class _AnalysisCSV(RawIOBase):
    """Replay a validated header in memory before borrowing the remaining source bytes."""

    def __init__(self, header: tuple[str, ...], source: IO[bytes]) -> None:
        self._prefix = (",".join(header) + "\n").encode()
        self._source = source

    @override
    def readable(self) -> bool:
        """Allow Arrow to consume the buffered header and remaining payload."""
        return True

    @override
    def read(self, size: int = -1, /) -> bytes:
        """Read at most the requested number of bytes in source order."""
        if not self._prefix:
            return self._source.read(size)
        if size < 0:
            prefix, self._prefix = self._prefix, b""
            return prefix + self._source.read(size)
        prefix, self._prefix = self._prefix[:size], self._prefix[size:]
        return prefix if len(prefix) == size else prefix + self._source.read(size - len(prefix))


def _identity(name: str, seen: set[tuple[str, str]], path: Path, context: ReaderErrorContext) -> tuple[str, str] | None:
    """Reject ambiguous semantic artifacts before reading any selected payload."""
    match = _MEMBER.fullmatch(name)
    if match is None or match["file"] != _FILES[match["kind"]]:
        return None
    identity = match["kind"], match["method"]
    if (match["kind"], match["method"].lower()) in seen:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"repeated analysis identity {identity!r}",
            expected="one CSV per kind and method, ignoring letter case",
        )
    seen.add((match["kind"], match["method"].lower()))
    return identity


def _members(
    source: AnalysisSource, context: ReaderErrorContext
) -> Generator[tuple[int, str, str, IO[bytes], Path], None, None]:
    """Own selected CSV streams after rejecting repeated semantic identities."""
    seen: set[tuple[str, str]] = set()
    if isinstance(source, AnalysisDirectory):
        files: list[tuple[str, str, Path]] = []
        for relative, path in source.files:
            identity = _identity(relative.as_posix(), seen, path, context)
            if identity is not None:
                files.append((*identity, path))
        for order, (kind, method, path) in enumerate(files):
            with path.open("rb") as handle:
                yield order, kind, method, handle, path
        return
    with tarfile.open(source.file, "r:gz") as archive:
        members: list[tuple[int, str, str, tarfile.TarInfo]] = []
        ordered = sorted((member for member in archive if member.isfile()), key=lambda member: member.name)
        for order, member in enumerate(ordered):
            identity = _identity(str(PurePosixPath(member.name)), seen, source.file, context)
            if identity is not None:
                members.append((order, *identity, member))
        # Parse in physical order. The gzip stream needs one rewind after preflight;
        # the saved order determines the returned annotations.
        for order, kind, method, member in sorted(members, key=lambda item: item[3].offset):
            member_handle = archive.extractfile(member)
            if member_handle is None:  # pragma: no cover - isfile() members always have a payload
                raise ReaderFormatError(
                    context=context, path=source.file, found=f"{member.name} without payload", expected="a regular CSV"
                )
            with member_handle:
                yield order, kind, method, member_handle, source.file


def _table(
    handle: IO[bytes], header: tuple[str, ...], value: pa.DataType, path: Path, context: ReaderErrorContext
) -> pa.Table:
    """Parse an owned CSV stream with text barcodes and typed values."""
    # Keep Arrow's header-only, record numbering and first-record BOM semantics
    # without seeking backwards in a compressed archive.
    with _AnalysisCSV(header, handle) as stream, csv_errors(context=context, path=path):
        return pacsv.read_csv(
            stream,
            read_options=pacsv.ReadOptions(use_threads=False),
            convert_options=pacsv.ConvertOptions(
                column_types={BARCODE: pa.string(), **dict.fromkeys(header[1:], value)},
                include_columns=list(header),
                null_values=[],
                strings_can_be_null=False,
            ),
        )


def _rows(table: pa.Table, barcodes: pd.Index, path: Path, context: ReaderErrorContext) -> NDArray[np.intp]:
    """Return the cell row of every record; barcodes must be unique and known."""
    rows = barcodes.get_indexer(pd.Index(arrow_series(table[BARCODE])))
    if (rows < 0).any() or not pd.Index(rows).is_unique:
        raise ReaderFormatError(
            context=context, path=path, found="unknown or repeated barcodes", expected="unique barcodes of known cells"
        )
    return rows


def _clusters(table: pa.Table, barcodes: pd.Index, path: Path, context: ReaderErrorContext) -> pd.Categorical:
    values = table[CLUSTER].to_numpy()
    if (values == 0).any():
        raise ReaderFormatError(
            context=context, path=path, found="nonpositive clusters", expected="positive integer clusters"
        )
    categories = np.unique(values)
    codes = np.full(len(barcodes), -1, dtype=np.int64)
    codes[_rows(table, barcodes, path, context)] = np.searchsorted(categories, values)
    return pd.Categorical.from_codes(codes, categories=pd.Index(categories.astype(np.str_), dtype="str"))


def _projection(
    table: pa.Table, header: tuple[str, ...], barcodes: pd.Index, path: Path, context: ReaderErrorContext
) -> NDArray[np.float64]:
    # AnnData stores the projection as one NumPy obsm array.
    values = np.column_stack([table[name].to_numpy() for name in header[1:]])
    if not np.isfinite(values).all():
        raise ReaderFormatError(
            context=context, path=path, found="nonfinite projections", expected="finite projections"
        )
    matrix = np.full((len(barcodes), len(header) - 1), np.nan)
    matrix[_rows(table, barcodes, path, context)] = values
    return matrix


def read_analysis(source: AnalysisSource, barcodes: pd.Index, *, context: ReaderErrorContext) -> Analysis:
    """Align every clustering and projection to ``barcodes``; absent cells are missing.

    Raises
    ------
    ReaderFormatError
        If a CSV has an unexpected header, malformed or nonfinite values,
        nonpositive clusters, unknown or repeated barcodes, or repeated semantic
        artifacts.
    """
    clusters: dict[int, tuple[str, pd.Categorical]] = {}
    projections: dict[int, tuple[str, NDArray[np.float64]]] = {}
    with closing(_members(source, context)) as members:
        for order, kind, method, handle, path in members:
            header = csv_header(path, context=context, source=handle)
            validate_analysis_header(header, kind, path=path, context=context)
            if kind == "clustering":
                table = _table(handle, header, pa.uint64(), path, context)
                clusters[order] = (method, _clusters(table, barcodes, path, context))
            else:
                table = _table(handle, header, pa.float64(), path, context)
                projections[order] = (f"{kind}_{method}", _projection(table, header, barcodes, path, context))
    return Analysis(
        dict(clusters[order] for order in sorted(clusters)), dict(projections[order] for order in sorted(projections))
    )
