"""MERSCOPE exact count/ID schemas, metadata primitives, and lazy event annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.merscope._schemas import (
    FLOAT_DTYPE,
    STRING_DTYPE,
    blank_columns,
    boundary_columns,
    count_columns,
    integer_values,
    metadata_columns,
    normalize_boundary_ids,
    normalize_cell_ids,
    normalize_floats,
    normalize_metadata,
    normalize_transcript_partition,
    validate_output_annotations,
    validate_transcript_columns,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestCountColumns:
    """Treat the first source field as identity and preserve ordered features."""

    @pytest.mark.parametrize("first", ["cell", ""])
    def test_order(self, tmp_path: Path, first: str) -> None:
        assert count_columns([first, "B", "Blank-1", "A"], ArtifactFile(tmp_path / "count.csv")) == (
            "B",
            "Blank-1",
            "A",
        )

    @pytest.mark.parametrize("columns", [[], ["cell"], ["cell", ""], ["cell", "A", "A"], ["cell", "A\x00"]])
    def test_bad_headers(self, tmp_path: Path, columns: list[str]) -> None:
        with pytest.raises(ReaderFormatError):
            count_columns(columns, ArtifactFile(tmp_path / "count.csv"))


class TestBlankColumns:
    """Do not classify genes by a substring or case folding."""

    def test_exact_names(self) -> None:
        assert blank_columns(("Blank-1", "Blank-20", "Blank-0", "blank-2", "BlanketGene", "Blank-01", "Gene")) == (
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        )


class TestNormalizeCellIds:
    """Retain opaque text identities and exact integer representations."""

    def test_lexical_and_integer(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "cells.csv")
        result = normalize_cell_ids(pd.Series(["001", "cell-a", str(2**63 + 1)]), path)
        assert result.tolist() == ["001", "cell-a", str(2**63 + 1)]
        assert result.dtype == STRING_DTYPE
        assert normalize_cell_ids(pd.Series([2**63 + 1], dtype="uint64"), path).tolist() == [str(2**63 + 1)]
        assert normalize_cell_ids(pd.Series(["-1"]), path, allow_unassigned=True).tolist() == ["-1"]

    @pytest.mark.parametrize("value", [None, "", "a\x00", "-1", 1.0, True])
    def test_invalid(self, tmp_path: Path, *, value: str | float | bool | None) -> None:
        with pytest.raises(ReaderFormatError):
            normalize_cell_ids(pd.Series([value]), ArtifactFile(tmp_path / "cells.csv"))


class TestIntegerValues:
    """Keep exact counts through parser spellings and integer-width promotion."""

    def test_normal_and_unsigned(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "count.csv")
        assert integer_values(pd.Series(["0", "3"], dtype=STRING_DTYPE), path).tolist() == [0, 3]
        result = integer_values(pd.Series([str(2**64 - 1), "2.0", "3e0"], dtype=STRING_DTYPE), path)
        assert result.tolist() == [2**64 - 1, 2, 3]
        assert result.dtype == np.dtype(np.uint64)
        assert integer_values(pd.Series([], dtype=STRING_DTYPE), path).dtype == np.dtype(np.int64)

    @pytest.mark.parametrize(
        "value", [None, "", "1.00000000000000001", "1e-400", "-1", "nan", "inf", str(2**64), "x", "1" * 129]
    )
    def test_invalid(self, tmp_path: Path, value: str | None) -> None:
        with pytest.raises(ReaderFormatError, match=r"integral|integer"):
            integer_values(pd.Series([value], dtype=STRING_DTYPE), ArtifactFile(tmp_path / "count.csv"))


class TestNormalizeFloats:
    """Keep correctly rounded float64 coordinates and reject destructive decoding."""

    def test_precision(self, tmp_path: Path) -> None:
        result = normalize_floats(
            pd.Series(["0.10000000000000002", "-0.00e1000", str(2**53)], index=[2, 4, 6]),
            ArtifactFile(tmp_path / "tx.csv"),
            "x",
        )
        assert result.dtype == FLOAT_DTYPE
        assert result.index.tolist() == [2, 4, 6]
        np.testing.assert_array_equal(result.to_numpy(dtype=np.float64), [0.10000000000000002, 0, 2**53])

    @pytest.mark.parametrize("value", [None, "", "nan", "inf", "1e400", "1e-400", str(2**53 + 1), "1" * 129])
    def test_invalid(self, tmp_path: Path, value: str | None) -> None:
        with pytest.raises(ReaderFormatError, match="float64"):
            normalize_floats(pd.Series([value], dtype=STRING_DTYPE), ArtifactFile(tmp_path / "tx.csv"), "x")


class TestMetadataColumns:
    """Accept additional named annotations independently of source column order."""

    def test_extra_fields(self, tmp_path: Path) -> None:
        assert metadata_columns(
            ["annotation", "center_y", "EntityID", "center_x"], ArtifactFile(tmp_path / "metadata.csv")
        ) == ("annotation", "center_y", "EntityID", "center_x")

    def test_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="metadata"):
            metadata_columns(["EntityID"], ArtifactFile(tmp_path / "metadata.csv"))


class TestNormalizeMetadata:
    """Interpret known physical fields and preserve extra annotation spellings."""

    def test_typed_fields_and_nullable_fov(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {
                "EntityID": ["a", "b"],
                "center_x": ["1", "2"],
                "center_y": ["3", "4"],
                "volume": ["0", "5"],
                "fov": ["", "2"],
                "transcript_count": ["0", str(2**63)],
                "extra": ["0001", "NA"],
            },
            dtype=STRING_DTYPE,
        )
        result = normalize_metadata(frame, ArtifactFile(tmp_path / "metadata.csv"))
        assert result.index.tolist() == ["a", "b"]
        assert result["center_x"].dtype == FLOAT_DTYPE
        assert result["extra"].tolist() == ["0001", "NA"]
        assert result["fov"].isna().tolist() == [True, False]
        assert result["transcript_count"].tolist() == [0, 2**63]

    @pytest.mark.parametrize(
        ("column", "values"), [("EntityID", ["a", "a"]), ("volume", ["-1", "2"]), ("fov", [str(2**63), "2"])]
    )
    def test_invalid(self, tmp_path: Path, column: str, values: list[str]) -> None:
        frame = pd.DataFrame(
            {"EntityID": ["a", "b"], "center_x": ["1", "2"], "center_y": ["3", "4"], column: values}, dtype=STRING_DTYPE
        )
        with pytest.raises(ReaderFormatError):
            normalize_metadata(frame, ArtifactFile(tmp_path / "metadata.csv"))


class TestBoundaryColumns:
    """Project only the geometry contract and optional entity type."""

    def test_projection(self, tmp_path: Path) -> None:
        assert boundary_columns(
            ["extra", "Geometry", "Type", "EntityID", "ZIndex"], ArtifactFile(tmp_path / "cells.parquet")
        ) == ("EntityID", "ZIndex", "Geometry", "Type")
        assert boundary_columns(["Geometry", "EntityID", "ZIndex"], ArtifactFile(tmp_path / "cells.parquet")) == (
            "EntityID",
            "ZIndex",
            "Geometry",
        )

    def test_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError):
            boundary_columns(["EntityID"], ArtifactFile(tmp_path / "cells.parquet"))


class TestValidateOutputAnnotations:
    """Prevent silent replacement of source annotations by reserved output fields."""

    @pytest.mark.parametrize("column", ["region", "slide", "dataset_id", "cells_region"])
    @pytest.mark.parametrize("value", ["conflict", None])
    def test_conflicts(self, column: str, value: str | None) -> None:
        frame = pd.DataFrame({column: [value]}, dtype=STRING_DTYPE)
        with pytest.raises(ReaderFormatError, match="conflicting source"):
            validate_output_annotations(frame, {column: "expected"}, context=ReaderErrorContext("merscope", "table"))


class TestNormalizeBoundaryIds:
    """Validate physical primitive types before WKB geometry decoding."""

    def test_repeated_planes_and_large_ids(self, tmp_path: Path) -> None:
        table = pa.table({"EntityID": [2**63 - 1, 2**63 - 1], "ZIndex": [0, 0], "Type": ["cell", "Cell"]})
        result = normalize_boundary_ids(table, ArtifactFile(tmp_path / "cells.parquet"))
        assert result.tolist() == [str(2**63 - 1)] * 2
        assert result.name == "EntityID"

    @pytest.mark.parametrize(
        "table",
        [
            pa.table({"EntityID": [1.0], "ZIndex": [0]}),
            pa.table({"EntityID": ["1"], "ZIndex": [str(2**63)]}),
            pa.table({"EntityID": ["1"], "ZIndex": [0], "Type": ["Nucleus"]}),
            pa.table({"EntityID": ["1"], "ZIndex": [0], "Type": pa.array([None], type=pa.string())}),
            pa.table({"EntityID": pa.array([], type=pa.string()), "ZIndex": pa.array([], type=pa.int64())}),
        ],
    )
    def test_invalid(self, tmp_path: Path, table: pa.Table) -> None:
        with pytest.raises(ReaderFormatError):
            normalize_boundary_ids(table, ArtifactFile(tmp_path / "cells.parquet"))


class TestValidateTranscriptColumns:
    """Project global coordinates and source annotations without local XY collisions."""

    def test_optional_projection(self, tmp_path: Path) -> None:
        columns = [
            "",
            "x",
            "y",
            "global_y",
            "gene",
            "global_x",
            "global_z",
            "fov",
            "barcode_id",
            "transcript_id",
            "cell_id",
        ]
        assert validate_transcript_columns(columns, ArtifactFile(tmp_path / "tx.csv")) == (
            "gene",
            "global_x",
            "global_y",
            "global_z",
            "cell_id",
            "fov",
            "barcode_id",
            "transcript_id",
        )

    def test_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError):
            validate_transcript_columns(["gene", "x", "y"], ArtifactFile(tmp_path / "tx.csv"))


class TestNormalizeTranscriptPartition:
    """Validate only executed events and preserve assignment provenance."""

    @pytest.mark.parametrize("assignment", ["current", "original", "none"])
    def test_output(self, tmp_path: Path, assignment: Literal["current", "original", "none"]) -> None:
        frame = pd.DataFrame(
            {
                "gene": ["B", "A"],
                "global_x": [".10000000000000002", "2"],
                "global_y": ["3", "4"],
                "global_z": ["0.0", "2"],
                "cell_id": ["-1", str(2**63 + 1)],
                "fov": ["0", "1"],
                "barcode_id": ["2", "3"],
                "transcript_id": ["ENST001.3", "123"],
            },
            dtype=STRING_DTYPE,
        )
        result = normalize_transcript_partition(
            frame,
            features=("A", "B", "Blank-1"),
            path=ArtifactFile(tmp_path / "tx.csv"),
            assignment=assignment,
            allowed_cell_ids=(str(2**63 + 1),),
        )
        assert result["gene"].cat.categories.tolist() == ["A", "B", "Blank-1"]
        assert result["x"].dtype == np.dtype(np.float64)
        assert result["x"].iloc[0] == 0.10000000000000002
        assert result["z_raw"].tolist() == [0, 2]
        assert result["original_cell_id" if assignment == "original" else "cell_id"].tolist() == ["-1", str(2**63 + 1)]
        assert result["transcript_id"].tolist() == ["ENST001.3", "123"]

    @pytest.mark.parametrize(
        ("column", "value"),
        [("gene", "Unknown"), ("gene", None), ("global_z", str(2**63)), ("transcript_id", ""), ("cell_id", "other")],
    )
    def test_invalid(self, tmp_path: Path, column: str, value: str | None) -> None:
        frame = pd.DataFrame({"gene": ["A"], "global_x": ["1"], "global_y": ["2"], column: [value]}, dtype=STRING_DTYPE)
        with pytest.raises(ReaderFormatError):
            normalize_transcript_partition(
                frame,
                features=("A",),
                path=ArtifactFile(tmp_path / "tx.csv"),
                assignment="current",
                allowed_cell_ids=("a",),
            )

    def test_empty(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({name: pd.Series(dtype=STRING_DTYPE) for name in ("gene", "global_x", "global_y")})
        assert normalize_transcript_partition(
            frame, features=("A",), path=ArtifactFile(tmp_path / "tx.csv"), assignment="none"
        ).empty
