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
    _FLOAT_DTYPE,
    GEOMETRY_REPRESENTATION_DTYPE,
    STRING_DTYPE,
    blank_columns,
    boundary_columns,
    count_columns,
    count_types,
    metadata_columns,
    metadata_types,
    normalize_boundary_ids,
    normalize_cell_ids,
    normalize_metadata,
    normalize_transcript_partition,
    require_finite,
    transcript_types,
    validate_output_annotations,
    validate_transcript_columns,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestGeometryRepresentation:
    """Keep literal-component and union semantics explicit in categorical output."""

    def test_categories_are_stable_and_unordered(self) -> None:
        assert GEOMETRY_REPRESENTATION_DTYPE.categories.tolist() == ["union_footprint", "source_components"]
        assert not GEOMETRY_REPRESENTATION_DTYPE.ordered


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
        result = normalize_cell_ids(pd.Series(["001", "cell-a", str(2**63 + 1)]), path, "cell metadata")
        assert result.tolist() == ["001", "cell-a", str(2**63 + 1)]
        assert result.dtype == STRING_DTYPE
        assert normalize_cell_ids(pd.Series([2**63 + 1], dtype="uint64"), path, "cell metadata").tolist() == [
            str(2**63 + 1)
        ]
        assert normalize_cell_ids(pd.Series(["-1"]), path, "transcripts", allow_unassigned=True).tolist() == ["-1"]

    @pytest.mark.parametrize("value", [None, "", "a\x00", "-1", 1.0, True])
    def test_invalid(self, tmp_path: Path, *, value: str | float | bool | None) -> None:
        with pytest.raises(ReaderFormatError):
            normalize_cell_ids(pd.Series([value]), ArtifactFile(tmp_path / "cells.csv"), "cell metadata")


class TestCountTypes:
    def test_unsigned_and_exact_identity(self, tmp_path: Path) -> None:
        types = count_types(["", "B", "A"], ArtifactFile(tmp_path / "counts.csv"))
        assert types == pa.schema([("", pa.string()), ("B", pa.uint64()), ("A", pa.uint64())])


class TestMetadataTypes:
    def test_roles(self, tmp_path: Path) -> None:
        assert metadata_types(
            ["EntityID", "center_x", "center_y", "fov", "transcript_count", "extra"],
            ArtifactFile(tmp_path / "metadata.csv"),
        ) == {
            "EntityID": pa.string(),
            "center_x": pa.float64(),
            "center_y": pa.float64(),
            "fov": pa.uint64(),
            "transcript_count": pa.uint64(),
            "extra": pa.string(),
        }


class TestTranscriptTypes:
    def test_roles(self, tmp_path: Path) -> None:
        assert transcript_types(
            ["gene", "global_x", "global_y", "global_z", "fov", "barcode_id", "cell_id", "transcript_id"],
            ArtifactFile(tmp_path / "events.csv"),
        ) == {
            "gene": pa.string(),
            "global_x": pa.float64(),
            "global_y": pa.float64(),
            "global_z": pa.uint64(),
            "fov": pa.uint64(),
            "barcode_id": pa.uint64(),
            "cell_id": pa.string(),
            "transcript_id": pa.string(),
        }


class TestRequireFinite:
    @pytest.mark.parametrize("value", [None, float("inf"), float("nan")])
    def test_invalid(self, tmp_path: Path, value: float | None) -> None:
        with pytest.raises(ReaderFormatError, match="missing or nonfinite"):
            require_finite(
                pd.Series([value], dtype=_FLOAT_DTYPE),
                ArtifactFile(tmp_path / "metadata.csv"),
                "cell metadata",
                "center_x",
            )


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
                "fov": [None, "2"],
                "transcript_count": ["0", str(2**63)],
                "extra": ["0001", "NA"],
            },
            dtype=STRING_DTYPE,
        )
        frame = (
            pa.Table.from_pandas(frame)
            .cast(
                pa.schema(list(metadata_types(frame.columns.tolist(), ArtifactFile(tmp_path / "metadata.csv")).items()))
            )
            .to_pandas(types_mapper=pd.ArrowDtype)
        )
        result = normalize_metadata(frame, ArtifactFile(tmp_path / "metadata.csv"))
        assert result.index.tolist() == ["a", "b"]
        assert result.index.name is None
        assert result["center_x"].dtype == _FLOAT_DTYPE
        assert result["extra"].tolist() == ["0001", "NA"]
        assert result["fov"].isna().tolist() == [True, False]
        assert result["transcript_count"].tolist() == [0, 2**63]

    @pytest.mark.parametrize(
        ("column", "values"), [("EntityID", ["a", "a"]), ("volume", ["-1", "2"]), ("transcript_count", [None, "2"])]
    )
    def test_invalid(self, tmp_path: Path, column: str, values: list[str | None]) -> None:
        frame = pd.DataFrame(
            {"EntityID": ["a", "b"], "center_x": ["1", "2"], "center_y": ["3", "4"], column: values}, dtype=STRING_DTYPE
        )
        types = metadata_types(frame.columns.tolist(), ArtifactFile(tmp_path / "metadata.csv"))
        frame = pa.Table.from_pandas(frame).cast(pa.schema(list(types.items()))).to_pandas(types_mapper=pd.ArrowDtype)
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

    @pytest.mark.parametrize("planes", [pa.array([None, 2], type=pa.int64()), pa.nulls(2)])
    def test_nullable_plane_annotations(self, tmp_path: Path, planes: pa.Array) -> None:
        table = pa.table({"EntityID": ["a", "a"], "ZIndex": planes, "Type": ["Cell", "Cell"]})
        assert normalize_boundary_ids(table, ArtifactFile(tmp_path / "cells.parquet")).tolist() == ["a", "a"]

    def test_uint64_planes(self, tmp_path: Path) -> None:
        table = pa.table({"EntityID": ["a"], "ZIndex": pa.array([2**64 - 1], pa.uint64())})
        assert normalize_boundary_ids(table, ArtifactFile(tmp_path / "cells.parquet")).tolist() == ["a"]

    def test_repeated_planes_and_large_ids(self, tmp_path: Path) -> None:
        table = pa.table({"EntityID": [2**63 - 1, 2**63 - 1], "ZIndex": [0, 0], "Type": ["cell", "Cell"]})
        result = normalize_boundary_ids(table, ArtifactFile(tmp_path / "cells.parquet"))
        assert result.tolist() == [str(2**63 - 1)] * 2
        assert result.name == "EntityID"

    @pytest.mark.parametrize(
        "table",
        [
            pa.table({"EntityID": [1.0], "ZIndex": [0]}),
            pa.table({"EntityID": ["1"], "ZIndex": [True]}),
            pa.table({"EntityID": ["1"], "ZIndex": [0.5]}),
            pa.table({"EntityID": ["1"], "ZIndex": ["-1"]}),
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

    @pytest.mark.parametrize("assignment", ["current", "original"])
    def test_output(self, tmp_path: Path, assignment: Literal["current", "original"]) -> None:
        frame = pd.DataFrame(
            {
                "gene": ["B", "A"],
                "global_x": [".10000000000000002", "2"],
                "global_y": ["3", "4"],
                "global_z": ["0", "2"],
                "cell_id": ["-1", str(2**63 + 1)],
                "fov": ["0", "1"],
                "barcode_id": ["2", "3"],
                "transcript_id": ["ENST001.3", "123"],
            },
            dtype=STRING_DTYPE,
        )
        types = transcript_types(frame.columns.tolist(), ArtifactFile(tmp_path / "tx.csv"))
        frame = pa.Table.from_pandas(frame).cast(pa.schema(list(types.items()))).to_pandas(types_mapper=pd.ArrowDtype)
        result = normalize_transcript_partition(
            frame,
            features=("A", "B", "Blank-1"),
            path=ArtifactFile(tmp_path / "tx.csv"),
            assignment=assignment,
            cell_population=pd.Index([str(2**63 + 1)], dtype=STRING_DTYPE),
        )
        assert result["gene"].cat.categories.tolist() == ["A", "B", "Blank-1"]
        assert result["x"].dtype == np.dtype(np.float64)
        assert result["x"].iloc[0] == 0.10000000000000002
        assert result["z_raw"].tolist() == [0, 2]
        assert result["original_cell_id" if assignment == "original" else "cell_id"].tolist() == ["-1", str(2**63 + 1)]
        assert result["transcript_id"].tolist() == ["ENST001.3", "123"]

    @pytest.mark.parametrize(
        ("column", "value"),
        [("gene", "Unknown"), ("gene", None), ("transcript_id", ""), ("cell_id", "other")],
    )
    def test_invalid(self, tmp_path: Path, column: str, value: str | None) -> None:
        frame = pd.DataFrame({"gene": ["A"], "global_x": ["1"], "global_y": ["2"], column: [value]}, dtype=STRING_DTYPE)
        types = transcript_types(frame.columns.tolist(), ArtifactFile(tmp_path / "tx.csv"))
        frame = pa.Table.from_pandas(frame).cast(pa.schema(list(types.items()))).to_pandas(types_mapper=pd.ArrowDtype)
        with pytest.raises(ReaderFormatError):
            normalize_transcript_partition(
                frame,
                features=("A",),
                path=ArtifactFile(tmp_path / "tx.csv"),
                assignment="current",
                cell_population=pd.Index(["a"], dtype=STRING_DTYPE),
            )

    def test_empty(self, tmp_path: Path) -> None:
        frame = pa.table(
            {
                "gene": pa.array([], pa.string()),
                "global_x": pa.array([], pa.float64()),
                "global_y": pa.array([], pa.float64()),
            }
        ).to_pandas(types_mapper=pd.ArrowDtype)
        assert normalize_transcript_partition(
            frame, features=("A",), path=ArtifactFile(tmp_path / "tx.csv"), assignment="current"
        ).empty
