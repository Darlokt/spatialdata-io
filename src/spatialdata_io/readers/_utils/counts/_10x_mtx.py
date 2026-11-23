"""Read sparse integer Matrix Market records and literal 10x MEX annotation fields."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from anndata import AnnData
from scipy import sparse

from spatialdata_io.readers._utils.csv import csv_errors, open_csv_stream
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import (
    as_csr_array,
    finalize_count_arrays,
    prepare_anndata_dataframe,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from typing import BinaryIO

    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile


@dataclass(frozen=True, slots=True)
class MtxFiles:
    """Validated matrix, barcode and feature paths in one 10x MEX family."""

    matrix: ArtifactFile
    barcodes: ArtifactFile
    features: ArtifactFile
    compressed: bool
    legacy: bool = False


def read_10x_mtx_fields(
    path: ArtifactFile, fields: Mapping[str, int], /, *, compressed: bool, context: ReaderErrorContext
) -> pd.DataFrame:
    """Read named projections of zero-based fields in a headerless MEX annotation TSV.

    UTF-8 fields retain their literal spelling and the returned Arrow-backed
    columns follow the order of the supplied field names.
    """
    types = {f"f{i}": pa.large_string() for i in fields.values()}
    try:
        with (
            open_csv_stream(path, context=context, compressed=compressed) as stream,
            csv_errors(context=context, path=path),
        ):
            if not stream.peek(1):
                return (
                    pa.schema({name: pa.large_string() for name in fields})
                    .empty_table()
                    .to_pandas(types_mapper=pd.ArrowDtype)
                )
            table = pacsv.read_csv(
                stream,
                read_options=pacsv.ReadOptions(autogenerate_column_names=True),
                parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                convert_options=pacsv.ConvertOptions(
                    column_types=types, include_columns=list(types), null_values=[], strings_can_be_null=False
                ),
            )
    except pa.ArrowKeyError as error:
        raise ReaderFormatError(
            context=context, path=path, found=str(error), expected=f"MEX annotation fields {dict(fields)}"
        ) from error
    return table.rename_columns(list(fields)).to_pandas(types_mapper=pd.ArrowDtype)


_BLOCK_BYTES = 4 * 1024 * 1024
_COORDINATE_FIELDS = 3
_DECIMAL_BYTES = b"0123456789- \r\n"


def _records(block: bytes) -> bytes:
    """Adapt Matrix Market whitespace, comments and decimal signs to Arrow's single delimiter."""
    if b"\t" in block:
        block = block.replace(b"\t", b" ")
    if b"%" in block:
        block = re.sub(rb"(?m)^ *%[^\n]*", b"", block)
    if b"+" in block:
        block = re.sub(rb"(?<!\S)\+(?=[0-9])", b"", block)
    if block.translate(None, _DECIMAL_BYTES):
        message = "Matrix Market records require decimal integer fields"
        raise ValueError(message)
    if b"  " in block:
        block = re.sub(rb" {2,}", b" ", block)
    if block.startswith(b" ") or block.endswith(b" ") or b"\n " in block or b" \n" in block or b" \r" in block:
        block = block.replace(b"\n ", b"\n").replace(b" \n", b"\n").replace(b" \r", b"\r").strip(b" ")
    return block


def _blocks(stream: BinaryIO) -> Iterator[bytes]:
    """Keep complete lines together, retaining at most two source blocks per read."""
    while block := stream.read(_BLOCK_BYTES):
        if not block.endswith(b"\n"):
            partial = len(block) - block.rfind(b"\n") - 1
            tail = stream.readline(_BLOCK_BYTES - partial + 1)
            if partial + len(tail) > _BLOCK_BYTES:
                message = "Matrix Market record exceeds the source block limit"
                raise ValueError(message)
            block += tail
        records = _records(block)
        if records and not records.isspace():
            yield records


def _dimensions(stream: BinaryIO) -> tuple[int, int, int]:
    while line := stream.readline(_BLOCK_BYTES):
        if len(line) == _BLOCK_BYTES and not line.endswith(b"\n"):
            message = "Matrix Market size or comment line exceeds the source block limit"
            raise ValueError(message)
        fields = line.split()
        if fields and not fields[0].startswith(b"%"):
            if len(fields) == _COORDINATE_FIELDS and all(field.removeprefix(b"+").isdigit() for field in fields):
                return int(fields[0]), int(fields[1]), int(fields[2])
            break
    message = "expected three nonnegative integer Matrix Market dimensions"
    raise ValueError(message)


def _coordinates(
    stream: BinaryIO, shape: tuple[int, int], entries: int
) -> tuple[NDArray[np.int32 | np.int64], NDArray[np.int32 | np.int64], NDArray[np.int64]] | None:
    index = pa.int32() if max(shape) <= np.iinfo(np.int32).max else pa.int64()
    types = {"feature": index, "barcode": index, "count": pa.int64()}
    read = pacsv.ReadOptions(column_names=list(types))
    parse = pacsv.ParseOptions(delimiter=" ", quote_char=False)
    convert = pacsv.ConvertOptions(
        column_types=types, include_columns=list(types), null_values=[], strings_can_be_null=False
    )
    tables = []
    seen = 0
    for block in _blocks(stream):
        table = pacsv.read_csv(pa.py_buffer(block), read_options=read, parse_options=parse, convert_options=convert)
        seen += table.num_rows
        if seen > entries:
            message = f"Matrix Market declares {entries} records but contains more"
            raise ValueError(message)
        tables.append(table)
    if seen != entries:
        message = f"expected {entries} Matrix Market records, found {seen}"
        raise ValueError(message)
    if not tables:
        return None
    rows, columns, data = pa.concat_tables(tables).columns
    return rows.to_numpy(), columns.to_numpy(), data.to_numpy()


def _mtx_counts(
    path: ArtifactFile, shape: tuple[int, int], *, compressed: bool, context: ReaderErrorContext
) -> sparse.csr_array:
    with open_csv_stream(path, context=context, compressed=compressed) as stream:
        banner = stream.readline(_BLOCK_BYTES)
        header = banner.lower().split()
        if (
            not banner.endswith(b"\n")
            or header[:-1] != [b"%%matrixmarket", b"matrix", b"coordinate", b"integer"]
            or header[-1:]
            not in (
                [b"general"],
                [b"symmetric"],
            )
        ):
            raise ReaderFormatError(
                context=context,
                path=path,
                found=f"MatrixMarket header {banner[:128]!r}",
                expected="an integer coordinate matrix with general or symmetric storage",
            )
        features, barcodes, entries = _dimensions(stream)
        symmetric = header[-1] == b"symmetric"
        if (features, barcodes) != shape or (symmetric and features != barcodes):
            message = f"Matrix Market dimensions {(features, barcodes)} do not match annotations {shape} or symmetry"
            raise ValueError(message)
        coordinates = _coordinates(stream, shape, entries)
    if coordinates is None:
        return sparse.csr_array((barcodes, features), dtype=np.float32)
    row, column, data = coordinates
    if row.min() < 1 or row.max() > features or column.min() < 1 or column.max() > barcodes:
        message = "Matrix Market coordinates outside the declared matrix dimensions"
        raise ValueError(message)
    if data.min() < 0:
        message = "negative MatrixMarket count"
        raise ValueError(message)
    if symmetric:
        if (row < column).any():
            message = "symmetric Matrix Market records must occupy the lower triangle"
            raise ValueError(message)
        off_diagonal = row != column
        row, column = np.concatenate((row, column[off_diagonal])), np.concatenate((column, row[off_diagonal]))
        data = np.concatenate((data, data[off_diagonal]))
    source = sparse.coo_array((data, (column - 1, row - 1)), shape=(barcodes, features))
    # SciPy's native conversion sums duplicate coordinates in the input dtype.
    # Use it when even concentrating all contributions in one cell is safe.
    counts = source.tocsr() if int(data.max()) * entries <= np.iinfo(np.int64).max else as_csr_array(source)
    return finalize_count_arrays((counts,), context=context, path=path)[0]


def read_10x_mtx(files: MtxFiles, /, *, context: ReaderErrorContext, make_unique: bool = True) -> AnnData:
    """Read an explicit 10x MEX triplet into an owned observation-by-feature table.

    The first two legacy feature fields are the ID and symbol. Modern MEX
    additionally supplies the feature type. Repeated matrix coordinates are
    additive and names are decoded literally. Only feature symbols are made
    unique when requested. All streams close before return.
    """
    barcodes = read_10x_mtx_fields(files.barcodes, {"barcode": 0}, compressed=files.compressed, context=context)
    fields = {"gene_ids": 0, "symbol": 1}
    if not files.legacy:
        fields["feature_types"] = 2
    var = read_10x_mtx_fields(files.features, fields, compressed=files.compressed, context=context)
    try:
        counts = _mtx_counts(files.matrix, (len(var), len(barcodes)), compressed=files.compressed, context=context)
    except ReaderFormatError:
        raise
    except (ValueError, OverflowError) as error:
        raise ReaderFormatError(
            context=context,
            path=files.matrix,
            found=str(error),
            expected="a well-formed 10x integer coordinate matrix aligned to the annotation TSVs",
        ) from error
    obs = pd.DataFrame(index=pd.Index(barcodes["barcode"].array))
    var.index = pd.Index(var.pop("symbol").array)
    table = AnnData(
        counts,
        obs=prepare_anndata_dataframe(obs, context=context),
        var=prepare_anndata_dataframe(var, context=context),
    )
    if make_unique:
        table.var_names_make_unique()
    return table
