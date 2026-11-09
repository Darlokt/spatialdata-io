"""Read Xenium counts and cell tables and assemble the one linked expression table.

Counts come from ``cell_feature_matrix.h5`` or its MatrixMarket twin (directory or
``cell_feature_matrix.tar.gz``); 10x writes identical values to both. Scanpy
normally returns float32 values, suitable for realistic count magnitudes. Protein
Expression features (XOA 4.0+) store the mean stain intensity multiplied by the
integer ``protein_scaling_factor`` of the HDF5 root attributes; the MatrixMarket
twin lacks that factor. Feature annotations become ``var`` columns; names
that a SpatialData table does not accept are renamed with a warning.

``cells.parquet`` (or its CSV twin) holds one row per cell in count-matrix
barcode order: ``cell_id``, ``x_centroid``/``y_centroid`` and ``cell_area``/
``nucleus_area`` in micrometres, per-cell transcript counts and, from XOA 2.0,
``segmentation_method``. Its columns become ``obs`` in source order. Column
names that a SpatialData table does not accept, including case variants of
the documented and reader-added columns, are renamed with a warning; the
documented columns keep their names. ``cells.zarr.zip`` adds ``label_id`` and
the nucleus centroids, and secondary-analysis clusterings are added last under
their method names.
"""

from __future__ import annotations

import gzip
import tarfile
from dataclasses import dataclass, replace
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
from spatialdata_io.readers._utils.csv import csv_errors, csv_header
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.metadata import decode_hdf5_text
from spatialdata_io.readers._utils.names import sanitize_column_names
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile, require_directory
from spatialdata_io.readers._utils.table import (
    TableLinkage,
    finalize_count_arrays,
    normalize_owned_anndata,
    parse_linked_table,
    prepare_anndata_dataframe,
)
from spatialdata_io.readers.xenium._layout import H5Counts, MtxDirectory
from spatialdata_io.readers.xenium._schemas import (
    CELL_COLUMNS,
    CELL_FLOATS,
    CELL_ID,
    CELL_OPTIONAL_COUNTS,
    CELL_REQUIRED,
    CELL_REQUIRED_COUNTS,
    READER_OBS,
    STRING,
    STRING_DTYPE,
    arrow_series,
    cast_column,
    header_types,
    id_index,
    normalize_ids,
    require_finite,
    validate_header,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy import sparse

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers.xenium._analysis import Analysis
    from spatialdata_io.readers.xenium._cells import CellStore
    from spatialdata_io.readers.xenium._layout import CountSource, TableTarget, TabularSource
    from spatialdata_io.readers.xenium._metadata import ProteinPanel

type FeatureSelection = tuple[str, ...] | Literal["all"] | None

_GENE_EXPRESSION = "Gene Expression"
_PROTEIN_EXPRESSION = "Protein Expression"
_PROTEIN_METRIC = "MEAN_PER_CELL_STAIN"
# var columns the reader relies on or adds by name.
_PANEL_COLUMNS = ("long_name", "clone", "codeword_index")
_VAR_COLUMNS = ("gene_ids", "feature_types", *_PANEL_COLUMNS)
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


def _source_path(source: CountSource) -> Path:
    return source.directory if isinstance(source, MtxDirectory) else source.file


def _extract_archive(
    file: ArtifactFile, destination: Path, *, context: ReaderErrorContext, metadata_only: bool = False
) -> ArtifactDirectory:
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
            raise ReaderFormatError(
                context=context,
                path=file,
                found=f"members {sorted(member.name for member in members)}",
                expected="exactly one cell_feature_matrix/{barcodes,features}.tsv.gz and matrix.mtx.gz",
            )
        # The data filter additionally rejects links and unsafe names.
        selected = [
            member for member in members if not metadata_only or PurePosixPath(member.name).name != "matrix.mtx.gz"
        ]
        archive.extractall(destination, members=selected, filter="data")
    return require_directory(destination / "cell_feature_matrix", context=context)


def _read_matrix(source: CountSource, context: ReaderErrorContext) -> ad.AnnData:
    """Read every feature of a count source into an owned canonical CSR table."""
    if isinstance(source, H5Counts):
        return read_10x_h5(source.file, context=context)
    if isinstance(source, MtxDirectory):
        return read_10x_mtx(source.directory, context=context, make_unique=False)
    # The returned table holds only in-memory copies; the directory is removed.
    with TemporaryDirectory(prefix="xenium-mtx-") as temporary:
        directory = _extract_archive(source.file, Path(temporary), context=context)
        return read_10x_mtx(directory, context=context, make_unique=False)


def _protein_scaling(source: CountSource, context: ReaderErrorContext) -> int:
    """Read and validate the HDF5 root protein attributes."""
    if not isinstance(source, H5Counts):
        raise ReaderFormatError(
            context=context,
            path=_source_path(source),
            found="Protein Expression features in a MatrixMarket matrix without the scaling factor",
            expected="cell_feature_matrix.h5, or feature_types without 'Protein Expression'",
        )
    with h5py.File(source.file, "r") as handle:
        factor: object = handle.attrs.get("protein_scaling_factor")
        metric: object = handle.attrs.get("protein_metric")
    if not isinstance(factor, int | np.integer) or isinstance(factor, bool) or factor <= 0:
        raise ReaderFormatError(
            context=context,
            path=source.file,
            found=f"protein_scaling_factor {factor!r}",
            expected="a positive integer protein_scaling_factor",
        )
    metric = decode_hdf5_text(metric, attribute="protein_metric", context=context, path=source.file)
    if metric != _PROTEIN_METRIC:
        raise ReaderFormatError(
            context=context, path=source.file, found=f"protein_metric {metric!r}", expected=repr(_PROTEIN_METRIC)
        )
    return int(factor)


def _selected_types(
    available: tuple[str, ...], feature_types: FeatureSelection, path: Path, context: ReaderErrorContext
) -> tuple[str, ...]:
    if feature_types == "all":
        return available
    if feature_types is None:
        return tuple(kind for kind in available if kind in (_GENE_EXPRESSION, _PROTEIN_EXPRESSION))
    missing = [kind for kind in feature_types if kind not in available]
    if missing:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"feature types {missing!r} absent",
            expected=f"feature_types from {list(available)!r}",
        )
    return feature_types


def _protein_annotations(
    var: pd.DataFrame, is_protein: NDArray[np.bool_], panel: ProteinPanel, path: Path, context: ReaderErrorContext
) -> None:
    """Add 1:1 protein panel annotations; other features stay missing."""
    conflicts = [name for name in _PANEL_COLUMNS if name in var]
    if conflicts:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"reserved protein annotation columns {conflicts!r}",
            expected="no source columns named long_name, clone or codeword_index when adding protein panel annotations",
        )
    ids = var["gene_ids"].to_numpy(dtype=np.str_)
    missing = [str(identifier) for identifier in ids[is_protein] if identifier not in panel.targets]
    if missing:
        raise ReaderFormatError(
            context=replace(context, artifact="protein_panel.json"),
            path=path,
            found=f"no targets for {missing!r}",
            expected="one target per protein feature",
        )
    targets = [
        panel.targets[identifier] if protein else None for identifier, protein in zip(ids, is_protein, strict=True)
    ]
    var["long_name"] = pd.Series([t and t.long_name for t in targets], index=var.index, dtype=STRING_DTYPE)
    var["clone"] = pd.Series([t and t.clone for t in targets], index=var.index, dtype=STRING_DTYPE)
    var["codeword_index"] = pd.Series(
        [t and t.codeword_index for t in targets], index=var.index, dtype=pd.ArrowDtype(pa.uint64())
    )


def read_counts(
    source: CountSource, feature_types: FeatureSelection, panel: ProteinPanel | None, *, context: ReaderErrorContext
) -> Counts:
    """Read, select and scale expression; source feature and barcode order are kept.

    Raises
    ------
    ReaderFormatError
        If feature names or barcodes repeat, count values are malformed,
        an explicit feature type is absent, protein is selected
        without a valid HDF5 factor and metric, a protein lacks a panel entry,
        or source columns conflict with the protein panel annotations.
    """
    source_table = _read_matrix(source, context)
    path = _source_path(source)
    source_var = cast("pd.DataFrame", source_table.var)
    if not source_table.var_names.is_unique or not source_table.obs_names.is_unique:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="repeated feature names or barcodes",
            expected="unique names and barcodes",
        )

    matrix = cast("sparse.csr_array", source_table.X)

    types = source_var["feature_types"].to_numpy(dtype=np.str_)
    selected = np.isin(types, _selected_types(tuple(dict.fromkeys(types)), feature_types, path, context))
    if not selected.all():
        matrix = matrix[:, selected]
    var = source_var.loc[selected]
    # Additional HDF5 feature datasets become var columns under their stored names.
    var.columns = sanitize_column_names(
        list(var.columns), source="xenium: the cell-feature matrix features", reserved=_VAR_COLUMNS
    )
    matrix = finalize_count_arrays((matrix,), context=context, path=path)[0]

    protein: dict[str, object] | None = None
    is_protein = types[selected] == _PROTEIN_EXPRESSION
    if is_protein.any():
        factor = _protein_scaling(source, context)
        entries = is_protein[matrix.indices]
        # Preserve the existing float32 divisor contract. Ordinary float32
        # tables divide once; exceptional exact counts retain float64 storage.
        matrix.data[entries] /= np.float32(factor)
        protein = {"scaling_factor": factor, "metric": _PROTEIN_METRIC}
        if panel is not None:
            _protein_annotations(var, is_protein, panel, path, context)

    table = ad.AnnData(X=matrix, obs=pd.DataFrame(index=source_table.obs_names), var=var)
    return Counts(table, source_table.var_names, protein)


def _tsv_column(payload: bytes, column: int, path: Path, context: ReaderErrorContext) -> pd.Index:
    """Return one column of a headerless gzip TSV of the MatrixMarket metadata as text."""
    name = f"f{column}"
    with csv_errors(context=context, path=path):
        table = pacsv.read_csv(
            pa.py_buffer(gzip.decompress(payload)),
            read_options=pacsv.ReadOptions(autogenerate_column_names=True),
            parse_options=pacsv.ParseOptions(delimiter="\t"),
            convert_options=pacsv.ConvertOptions(
                column_types={name: STRING}, include_columns=[name], null_values=[], strings_can_be_null=False
            ),
        )
    return pd.Index(arrow_series(table.column(0)), dtype="str")


def read_feature_index(source: CountSource, *, context: ReaderErrorContext) -> FeatureIndex:
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
            directory = _extract_archive(source.file, Path(temporary), context=context, metadata_only=True)
            features = (directory / "features.tsv.gz").read_bytes()
            barcodes_file = (directory / "barcodes.tsv.gz").read_bytes()
    path = _source_path(source)
    return FeatureIndex(_tsv_column(features, 1, path, context), _tsv_column(barcodes_file, 0, path, context))


def read_header(source: TabularSource, *, context: ReaderErrorContext) -> tuple[str, ...]:
    """Read only the header record of a CSV twin."""
    if source.kind == "csv":
        return csv_header(source.file, context=context)
    with gzip.open(source.file, "rb") as stream:
        return csv_header(source.file, context=context, source=stream)


def open_twin(source: TabularSource) -> pa.NativeFile:
    """Open a CSV twin for Arrow, decompressing a gzip twin as it is read."""
    return pa.input_stream(source.file, compression="gzip" if source.kind == "csv_gz" else None)


def read_cells(source: TabularSource, *, context: ReaderErrorContext) -> Cells:
    """Read ``cells.parquet`` or its CSV twin as validated, Arrow-backed annotations.

    Columns are cast to their declared types; a CSV twin is parsed into them
    directly, with an empty field as the missing value and other columns as
    text.

    Raises
    ------
    ReaderFormatError
        If columns are missing or repeated, the CSV is malformed, identifiers
        are missing or repeated, centroids or ``cell_area`` are not finite,
        areas or counts are negative, or ``nucleus_area`` is infinite (NaN
        marks cells without a nucleus).
    """
    path = source.file
    if source.kind == "parquet":
        table = pq.read_table(path)
        validate_header(table.column_names, CELL_REQUIRED, path=path, context=context)
    else:
        header = read_header(source, context=context)
        validate_header(header, CELL_REQUIRED, path=path, context=context)
        with csv_errors(context=context, path=path), open_twin(source) as stream:
            table = pacsv.read_csv(
                stream,
                convert_options=pacsv.ConvertOptions(
                    column_types=header_types(header, source.types),
                    include_columns=list(header),
                    null_values=[""],
                    strings_can_be_null=False,
                ),
            )

    names, types = table.column_names, source.types
    for name in (*CELL_FLOATS, *CELL_REQUIRED_COUNTS, *CELL_OPTIONAL_COUNTS):
        if name in names:
            column = cast_column(table[name], types.field(name).type, name, path=path, context=context)
            if name == "nucleus_area":
                # Cells without a nucleus have no area: fastparquet-written files
                # store null, Polars-written files and CSV twins NaN. Both become NaN.
                column = column.fill_null(float("nan"))
            table = table.set_column(names.index(name), name, column)
    ids = normalize_ids(table[CELL_ID], types.field(CELL_ID).type, path=path, context=context)
    table = table.set_column(names.index(CELL_ID), CELL_ID, ids)

    if any(table[name].null_count for name in CELL_REQUIRED):
        raise ReaderFormatError(
            context=context, path=path, found="missing required values", expected="complete cell records"
        )
    for name in CELL_FLOATS:
        values = table[name].to_numpy()
        require_finite(values, name, path=path, context=context, allow_nan=name == "nucleus_area")
        # NaN comparisons are false, so cells without a nucleus pass.
        if name in ("cell_area", "nucleus_area") and (values < 0).any():
            raise ReaderFormatError(
                context=context, path=path, found=f"negative {name} values", expected=f"nonnegative {name}"
            )

    ids = id_index(table[CELL_ID])
    if not ids.is_unique:
        raise ReaderFormatError(context=context, path=path, found="repeated cell IDs", expected="one row per cell")
    return Cells(table.to_pandas(types_mapper=pd.ArrowDtype), ids)


def build_table(  # noqa: PLR0913, RUF100 - Every table source and the error origin.
    cells: Cells,
    counts: Counts,
    store: CellStore | None,
    analysis: Analysis | None,
    *,
    path: Path,
    context: ReaderErrorContext,
) -> ad.AnnData:
    """Combine counts, cell annotations and optional analysis after exact identity checks.

    Raises
    ------
    ReaderFormatError
        If cell IDs differ from the count barcodes or the cell store in value or
        order, reader-owned annotation names already exist, segmentation methods
        disagree with the store, or a clustering is named like an annotation.

    Warns
    -----
    UserWarning
        If cells-table columns or clustering methods are not usable SpatialData
        column names and are renamed.
    """

    def error(*, found: str, expected: str) -> ReaderFormatError:
        return ReaderFormatError(context=context, path=path, found=found, expected=expected)

    barcodes = counts.table.obs_names
    # Integer IDs (XOA 1.0-1.2) are written as canonical decimal barcodes.
    if not cells.ids.astype("str").equals(barcodes.astype("str")):
        raise error(found="cell IDs differing from the count barcodes", expected="identical IDs in order")
    # The Zarr store and Arrow table have different index storage types; linkage
    # requires identical identifier values and order, not identical backends.
    if store is not None and (len(store.cell_ids) != len(cells.ids) or not (store.cell_ids == cells.ids).all()):
        raise error(found="cell IDs differing from cells.zarr.zip", expected="identical IDs in order")
    conflicts = [name for name in READER_OBS if name in cells.frame]
    if conflicts:
        raise error(found=f"reserved columns {conflicts!r}", expected="no reader-owned names")

    obs = cells.frame.set_axis(barcodes)
    obs.columns = sanitize_column_names(
        list(obs.columns), source="xenium: the cells table header", reserved=(*CELL_COLUMNS, *READER_OBS)
    )
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
                raise error(found="segmentation_method differing from cells.zarr.zip", expected="equal methods")
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
    if analysis is not None:
        _add_analysis(table, analysis, path, context)
    return table


def _add_analysis(table: ad.AnnData, analysis: Analysis, path: Path, context: ReaderErrorContext) -> None:
    """Add clusterings as ``obs`` columns and projections as ``obsm`` arrays."""
    obs = cast("pd.DataFrame", table.obs)
    shadowed = [method for method in analysis.clusters if method in obs]
    if shadowed:
        raise ReaderFormatError(
            context=replace(context, artifact="analysis"),
            path=path,
            found=f"clusterings {shadowed!r} named like cell annotations",
            expected="distinct analysis and annotation names",
        )
    methods = sanitize_column_names(
        list(analysis.clusters), source="xenium: the analysis clustering methods", reserved=(*obs.columns, "region")
    )
    for name, values in zip(methods, analysis.clusters.values(), strict=True):
        obs[name] = values
    for name, values in analysis.projections.items():
        table.obsm[name] = values


def link_table(
    table: ad.AnnData, target: TableTarget | None, expected: pd.Index | None, *, context: ReaderErrorContext
) -> ad.AnnData:
    """Convert annotations once and link the final table to its target element."""
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
