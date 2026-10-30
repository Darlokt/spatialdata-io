"""Read Xenium counts and cell tables and assemble the one linked expression table.

Counts come from ``cell_feature_matrix.h5`` or its MatrixMarket twin (directory or
``cell_feature_matrix.tar.gz``); 10x writes identical values to both. Scanpy
normally returns float32 values, suitable for realistic count magnitudes. Protein
Expression features (XOA 4.0+) store the mean stain intensity multiplied by the
integer ``protein_scaling_factor`` of the HDF5 root attributes; the MatrixMarket
twin lacks that factor.
"""

from __future__ import annotations

import csv
import gzip
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, cast

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from spatialdata_io._positions import POSITIONS_KEY, PositionsMetadata
from spatialdata_io.readers._utils.counts import read_10x_h5, read_10x_mtx
from spatialdata_io.readers._utils.errors import ReaderErrorContext
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile, require_directory
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    normalize_owned_anndata,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.xenium._layout import H5Counts, MtxDirectory
from spatialdata_io.readers.xenium._schemas import (
    CELL_FLOATS,
    CELL_ID,
    CELL_OPTIONAL_COUNTS,
    CELL_REQUIRED,
    CELL_REQUIRED_COUNTS,
    cast_column,
    cell_types,
    id_index,
    normalize_ids,
    require_finite,
    schema_error,
    validate_header,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray
    from scipy import sparse

    from spatialdata_io.readers.xenium._cells import CellStore
    from spatialdata_io.readers.xenium._layout import CountSource, TableTarget, TabularSource
    from spatialdata_io.readers.xenium._metadata import ProteinPanel
    from spatialdata_io.readers.xenium._version import Family

type FeatureSelection = tuple[str, ...] | Literal["all"] | None

GENE_EXPRESSION = "Gene Expression"
PROTEIN_EXPRESSION = "Protein Expression"
PROTEIN_METRIC = "MEAN_PER_CELL_STAIN"
RESERVED_OBS = ("label_id", "nucleus_centroid_x", "nucleus_centroid_y", "region")
STRING_DTYPE = pd.ArrowDtype(pa.string())
_MTX_MEMBERS = frozenset({"barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"})


@dataclass(frozen=True, slots=True)
class Cells:
    """Validated ``cells`` annotations in file order and their cell IDs."""

    frame: pd.DataFrame
    ids: pd.Index


@dataclass(frozen=True, slots=True)
class Counts:
    """Selected expression in source order and every source feature name.

    ``features`` lists all matrix features, the transcript categories. ``protein``
    records the applied protein scaling when protein is selected.
    """

    table: ad.AnnData
    features: pd.Index
    protein: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class FeatureIndex:
    """Feature names and barcodes of a count source, read without its values."""

    features: pd.Index
    barcodes: pd.Index


def _context(artifact: str) -> ReaderErrorContext:
    return ReaderErrorContext("xenium", artifact)


def _source_path(source: CountSource) -> Path:
    return source.directory if isinstance(source, MtxDirectory) else source.file


def _extract_archive(file: ArtifactFile, destination: Path, *, metadata_only: bool = False) -> ArtifactDirectory:
    """Validate required members once and extract only the selected, contained products."""
    with tarfile.open(file, "r:gz") as archive:
        members = [
            member
            for member in archive
            if member.isfile()
            and PurePosixPath(member.name).parent == PurePosixPath("cell_feature_matrix")
            and PurePosixPath(member.name).name in _MTX_MEMBERS
        ]
        names = [PurePosixPath(member.name).name for member in members]
        if set(names) != _MTX_MEMBERS or len(names) != len(_MTX_MEMBERS):
            raise schema_error(
                file,
                "cell_feature_matrix.tar.gz",
                found=f"members {sorted(member.name for member in members)}",
                expected="exactly one cell_feature_matrix/{barcodes,features}.tsv.gz and matrix.mtx.gz",
            )
        # The data filter additionally rejects links and unsafe names.
        selected = [
            member for member in members if not metadata_only or PurePosixPath(member.name).name != "matrix.mtx.gz"
        ]
        archive.extractall(destination, members=selected, filter="data")
    return require_directory(destination / "cell_feature_matrix", context=_context("cell_feature_matrix.tar.gz"))


def _read_matrix(source: CountSource) -> ad.AnnData:
    """Read every feature of a count source into an owned canonical CSR table."""
    if isinstance(source, H5Counts):
        return read_10x_h5(source.file, context=_context("cell_feature_matrix.h5"), gex_only=False)
    if isinstance(source, MtxDirectory):
        return read_10x_mtx(
            source.directory, context=_context("cell_feature_matrix"), make_unique=False, gex_only=False
        )
    # The returned table holds only in-memory copies; the directory is removed.
    with TemporaryDirectory(prefix="xenium-mtx-") as temporary:
        directory = _extract_archive(source.file, Path(temporary))
        return read_10x_mtx(
            directory, context=_context("cell_feature_matrix.tar.gz"), make_unique=False, gex_only=False
        )


def _protein_scaling(source: CountSource) -> int:
    """Read and validate the HDF5 root protein attributes."""
    if not isinstance(source, H5Counts):
        raise schema_error(
            _source_path(source),
            "protein scaling factor",
            found="Protein Expression features in a MatrixMarket matrix without the scaling factor",
            expected="cell_feature_matrix.h5, or feature_types without 'Protein Expression'",
        )
    with h5py.File(source.file, "r") as handle:
        factor: object = handle.attrs.get("protein_scaling_factor")
        metric: object = handle.attrs.get("protein_metric")
    metric = metric.decode() if isinstance(metric, bytes) else metric
    if not isinstance(factor, int | np.integer) or isinstance(factor, bool) or factor <= 0:
        raise schema_error(source.file, "protein_scaling_factor", found=f"{factor!r}", expected="a positive integer")
    if not np.isfinite(np.float32(factor)):
        raise schema_error(
            source.file, "protein_scaling_factor", found=f"{factor!r}", expected="a finite positive float32 divisor"
        )
    if metric != PROTEIN_METRIC:
        raise schema_error(source.file, "protein_metric", found=f"{metric!r}", expected=repr(PROTEIN_METRIC))
    return int(factor)


def _selected_types(available: tuple[str, ...], feature_types: FeatureSelection, path: Path) -> tuple[str, ...]:
    if feature_types == "all":
        return available
    if feature_types is None:
        return tuple(kind for kind in available if kind in (GENE_EXPRESSION, PROTEIN_EXPRESSION))
    missing = [kind for kind in feature_types if kind not in available]
    if missing:
        raise schema_error(
            path, "feature_types", found=f"feature types {missing!r} absent", expected=f"types from {list(available)!r}"
        )
    return feature_types


def _protein_annotations(var: pd.DataFrame, is_protein: NDArray[np.bool_], panel: ProteinPanel, path: Path) -> None:
    """Add 1:1 protein panel annotations; other features stay missing."""
    ids = var["gene_ids"].to_numpy(dtype=np.str_)
    missing = [str(identifier) for identifier in ids[is_protein] if identifier not in panel.targets]
    if missing:
        raise schema_error(
            path, "protein_panel.json", found=f"no targets for {missing!r}", expected="one target per protein feature"
        )
    targets = [
        panel.targets[identifier] if protein else None for identifier, protein in zip(ids, is_protein, strict=True)
    ]
    var["long_name"] = pd.Series([t and t.long_name for t in targets], index=var.index, dtype=STRING_DTYPE)
    var["clone"] = pd.Series([t and t.clone for t in targets], index=var.index, dtype=STRING_DTYPE)
    var["codeword_index"] = pd.Series(
        [t and t.codeword_index for t in targets], index=var.index, dtype=pd.ArrowDtype(pa.int64())
    )


def read_counts(source: CountSource, feature_types: FeatureSelection, panel: ProteinPanel | None) -> Counts:
    """Read, select and scale expression; source feature and barcode order are kept.

    Raises
    ------
    ReaderFormatError
        If feature names or barcodes repeat, count values are malformed,
        an explicit feature type is absent, protein is selected
        without a valid HDF5 factor and metric, or a protein lacks a panel entry.
    """
    source_table = _read_matrix(source)
    path = _source_path(source)
    source_var = cast("pd.DataFrame", source_table.var)
    if not source_table.var_names.is_unique or not source_table.obs_names.is_unique:
        raise schema_error(
            path,
            "cell-feature matrix",
            found="repeated feature names or barcodes",
            expected="unique names and barcodes",
        )

    matrix = cast("sparse.csr_array", source_table.X)

    types = source_var["feature_types"].to_numpy(dtype=np.str_)
    selected = np.isin(types, _selected_types(tuple(dict.fromkeys(types)), feature_types, path))
    if not selected.all():
        matrix = matrix[:, selected]
    var = source_var.loc[selected, ["gene_ids", "feature_types"]]

    protein: dict[str, object] | None = None
    is_protein = types[selected] == PROTEIN_EXPRESSION
    if is_protein.any():
        factor = _protein_scaling(source)
        entries = is_protein[matrix.indices]
        # One correctly rounded float32 division of exact operands; a float64
        # detour would round twice.
        matrix.data[entries] /= np.float32(factor)
        protein = {"scaling_factor": factor, "metric": PROTEIN_METRIC}
        if panel is not None:
            _protein_annotations(var, is_protein, panel, path)

    table = ad.AnnData(X=matrix, obs=pd.DataFrame(index=source_table.obs_names), var=var)
    return Counts(table, source_table.var_names, protein)


def _tsv_column(payload: bytes, column: int) -> pd.Index:
    rows = csv.reader(io.StringIO(gzip.decompress(payload).decode("utf-8")), delimiter="\t", strict=True)
    return pd.Index([row[column] for row in rows], dtype="str")


def read_feature_index(source: CountSource) -> FeatureIndex:
    """Read feature names and barcodes without the matrix values."""
    if isinstance(source, H5Counts):
        with h5py.File(source.file, "r") as handle:
            names = np.asarray(handle["matrix/features/name"][...]).astype(np.str_)
            barcodes = np.asarray(handle["matrix/barcodes"][...]).astype(np.str_)
        return FeatureIndex(pd.Index(names, dtype="str"), pd.Index(barcodes, dtype="str"))
    if isinstance(source, MtxDirectory):
        features = (source.directory / "features.tsv.gz").read_bytes()
        barcodes_file = (source.directory / "barcodes.tsv.gz").read_bytes()
    else:
        with TemporaryDirectory(prefix="xenium-mtx-") as temporary:
            directory = _extract_archive(source.file, Path(temporary), metadata_only=True)
            features = (directory / "features.tsv.gz").read_bytes()
            barcodes_file = (directory / "barcodes.tsv.gz").read_bytes()
    return FeatureIndex(_tsv_column(features, 1), _tsv_column(barcodes_file, 0))


def read_header(source: TabularSource) -> list[str]:
    """Read only the first record of a CSV twin."""
    if source.kind == "csv_gz":
        with gzip.open(source.file, "rt", encoding="utf-8-sig", newline="") as handle:
            line = handle.readline()
    else:
        with source.file.open(encoding="utf-8-sig", newline="") as handle:
            line = handle.readline()
    try:
        return next(csv.reader([line], strict=True), [])
    except csv.Error as error:
        raise schema_error(
            source.file, source.file.name, found="a malformed CSV header", expected="one CSV header record"
        ) from error


def read_csv_table(
    source: TabularSource,
    names: list[str],
    types: Mapping[str, pa.DataType],
    *,
    columns: Sequence[str] | None = None,
) -> pa.Table:
    """Read a CSV twin with its validated header and explicit types; empty fields are never null text.

    ``columns`` projects the result; Arrow still checks every record's full width
    but accepts some malformed quoting. Numbers are parsed directly into their
    declared float32 or float64 type, so each is rounded once.
    """
    try:
        with pa.input_stream(source.file, compression="gzip" if source.kind == "csv_gz" else None) as stream:
            return pacsv.read_csv(
                stream,
                read_options=pacsv.ReadOptions(column_names=names, skip_rows=1, use_threads=False),
                convert_options=pacsv.ConvertOptions(
                    include_columns=list(names if columns is None else columns),
                    column_types={name: types[name] for name in names if name in types},
                    strings_can_be_null=False,
                    quoted_strings_can_be_null=False,
                ),
            )
    except pa.ArrowInvalid as error:
        raise schema_error(
            source.file, source.file.name, found=f"malformed CSV: {error}", expected="typed records of the header width"
        ) from error


def read_cells(source: TabularSource, family: Family) -> Cells:
    """Read ``cells.parquet`` or its CSV twin as validated, Arrow-backed annotations.

    Raises
    ------
    ReaderFormatError
        If columns are missing or repeated, identifiers are missing or repeated,
        centroids or ``cell_area`` are not finite, areas or counts are negative,
        or ``nucleus_area`` is infinite (NaN marks cells without a nucleus).
    """
    path, artifact = source.file, source.file.name
    if source.kind == "parquet":
        table = pq.read_table(path)
        names = table.column_names
        validate_header(names, CELL_REQUIRED, path=path, artifact=artifact)
    else:
        names = read_header(source)
        validate_header(names, CELL_REQUIRED, path=path, artifact=artifact)
        table = read_csv_table(source, names, cell_types(names, family))

    types = cell_types(names, family)
    for name in (*CELL_FLOATS, *CELL_REQUIRED_COUNTS, *CELL_OPTIONAL_COUNTS):
        if name in names:
            column = cast_column(table[name], types[name], name, path=path, artifact=artifact)
            if name == "nucleus_area":
                # Cells without a nucleus have no area: fastparquet-written files
                # store null, Polars-written files and CSV twins NaN. Both become NaN.
                column = column.fill_null(float("nan"))
            table = table.set_column(names.index(name), name, column)
    table = table.set_column(
        names.index(CELL_ID), CELL_ID, normalize_ids(table[CELL_ID], family, path=path, artifact=artifact)
    )

    if any(table[name].null_count for name in CELL_REQUIRED):
        raise schema_error(path, artifact, found="missing required values", expected="complete cell records")
    for name in ("x_centroid", "y_centroid", "cell_area", "nucleus_area"):
        require_finite(table[name].to_numpy(), name, path=path, artifact=artifact, allow_nan=name == "nucleus_area")
    for name in (*CELL_REQUIRED_COUNTS, *CELL_OPTIONAL_COUNTS, "cell_area", "nucleus_area"):
        # NaN comparisons are false, so cells without a nucleus pass.
        if name in names and (np.asarray(table[name].to_numpy()) < 0).any():
            raise schema_error(path, artifact, found=f"negative {name} values", expected=f"nonnegative {name}")

    ids = id_index(table[CELL_ID])
    if not ids.is_unique:
        raise schema_error(path, artifact, found="repeated cell IDs", expected="one row per cell")
    return Cells(table.to_pandas(types_mapper=pd.ArrowDtype), ids)


def build_table(cells: Cells, counts: Counts, store: CellStore | None, *, path: Path) -> ad.AnnData:
    """Combine counts and cell annotations after exact, complete identity checks.

    Raises
    ------
    ReaderFormatError
        If cell IDs differ from the count barcodes or the cell store in value or
        order, reserved annotation names already exist, or segmentation methods
        disagree with the store.
    """
    barcodes = counts.table.obs_names
    # Integer IDs (XOA 1.0-1.2) are written as canonical decimal barcodes.
    if not cells.ids.astype("str").equals(barcodes.astype("str")):
        raise schema_error(
            path, "cells table", found="cell IDs differing from the count barcodes", expected="identical IDs in order"
        )
    if store is not None and not store.cell_ids.equals(cells.ids):
        raise schema_error(
            path, "cells.zarr.zip", found="cell IDs differing from the cells table", expected="identical IDs in order"
        )
    conflicts = [name for name in RESERVED_OBS if name in cells.frame]
    if conflicts:
        raise schema_error(
            path, "cells table", found=f"reserved columns {conflicts!r}", expected="no reader-owned names"
        )

    obs = cells.frame.set_axis(barcodes)
    if store is not None:
        obs["label_id"] = store.cell_labels
        obs["nucleus_centroid_x"] = store.nucleus_centroids[:, 0]
        obs["nucleus_centroid_y"] = store.nucleus_centroids[:, 1]
        if "nucleus_count" not in obs and store.nucleus_counts is not None:
            obs["nucleus_count"] = store.nucleus_counts
        if store.cell_methods is not None:
            if "segmentation_method" in obs and not np.array_equal(
                obs["segmentation_method"].to_numpy(dtype=np.str_), store.cell_methods
            ):
                raise schema_error(
                    path, "segmentation_method", found="methods differing from cells.zarr.zip", expected="equal methods"
                )
            obs["segmentation_method"] = pd.Series(store.cell_methods, index=obs.index, dtype=STRING_DTYPE)

    table = ad.AnnData(X=counts.table.X, obs=obs, var=cast("pd.DataFrame", counts.table.var))
    table.obsm["spatial"] = obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)
    positions: PositionsMetadata = {
        "schema_version": 1,
        "coordinates": {
            "spatial": {
                "axes": ["x", "y"],
                "units": "micrometer",
                "frame": "global",
                "position": "vendor cell centroid",
                "calibration": {"global": "Xenium cells x_centroid/y_centroid in micrometres"},
            }
        },
    }
    table.uns[POSITIONS_KEY] = positions
    return table


def link_table(table: ad.AnnData, target: TableTarget | None, expected: pd.Index | None) -> ad.AnnData:
    """Convert annotations once and link the final table to its target element."""
    context = ReaderErrorContext("xenium", "table")
    obs = cast("pd.DataFrame", table.obs)
    if target is not None:
        region: str = target
        obs["region"] = pd.Categorical.from_codes(np.zeros(len(obs), dtype=np.int8), categories=pd.Index([region]))
    prepare_anndata_dataframe(obs, context=context)
    prepare_anndata_dataframe(cast("pd.DataFrame", table.var), context=context)
    if target is None:
        return normalize_owned_anndata(table)
    return parse_linked_table(
        table,
        TableLinkage((target,), "region", "label_id" if target == "cell_labels" else CELL_ID),
        context=context,
        expected_instance_ids=None if expected is None else {target: expected},
    )
