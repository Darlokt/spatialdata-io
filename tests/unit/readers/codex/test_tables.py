"""CODEX layout, source schemas, identifier precision, and image options."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.codex import codex
from spatialdata_io.readers.codex._tables import _cell_columns, _cell_ids, _read_cell_table, _region_name
from tests.integration.readers.codex._fixtures import cell_frame, write_csv, write_fcs

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestCellColumns:
    """Keep vendor cycle channels in source order and metadata separate."""

    @pytest.mark.parametrize(
        "columns", [[], ["", "x"], ["x", "x"], ["cell_id", "region"], ["cell_id", "region", "x", "y"]]
    )
    def test_invalid_headers(self, tmp_path: Path, columns: list[str]) -> None:
        with pytest.raises(ReaderFormatError):
            _cell_columns(columns, ArtifactFile(tmp_path / "cells.csv"))

    def test_cycle_substring_in_metadata_is_not_expression(self, tmp_path: Path) -> None:
        features, metadata = _cell_columns(
            ["cell_id", "region", "x", "y", "cyc002_ch003", "recycle_score"], ArtifactFile(tmp_path / "cells.csv")
        )
        assert features == ["cyc002_ch003"]
        assert metadata == ["cell_id", "region", "x", "y", "recycle_score"]


class TestReadCellTable:
    """Exercise real source parser failures before geometry and linkage."""

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            ("cyc002_ch001", ["bad", "2", "3"]),
            ("cyc002_ch001", [np.inf, 2, 3]),
            ("x", [np.nan, 2, 3]),
            ("y", ["bad", "2", "3"]),
            ("cell_id", [7, 7, 11]),
            ("cell_id", ["bad", "2", "3"]),
            ("cell_id", [1.5, 2, 3]),
            ("cell_id", [np.nan, 2, 3]),
            ("cell_id", [np.inf, 2, 3]),
            ("cell_id", [str(2**64), "2", "3"]),
            ("cell_id", [str(-(2**63) - 1), "2", "3"]),
            ("cell_id", [str(2**63), "-1", "3"]),
            ("region", [1, 2, 1]),
            ("region", [np.nan, 1, 1]),
            ("region", [-1, -1, -1]),
            ("region", [1.5, 1.5, 1.5]),
            ("region", [np.inf, np.inf, np.inf]),
            ("x", [9007199254740993, 2, 3]),
        ],
    )
    def test_invalid_source_values(self, tmp_path: Path, column: str, values: list[str | int | float]) -> None:
        frame = cell_frame()
        frame[column] = values
        path = tmp_path / "cells.csv"
        write_csv(path, frame)
        with pytest.raises(
            (ReaderFormatError, ValueError), match=r"finite|cell_id|regions|region|numeric|precision|overflow|x/y"
        ):
            _read_cell_table(ArtifactFile(path))

    @pytest.mark.parametrize("source_format", ["csv", "fcs"])
    def test_empty_cells(self, tmp_path: Path, source_format: str) -> None:
        path = tmp_path / f"cells.{source_format}"
        if source_format == "csv":
            write_csv(path, cell_frame().iloc[:0])
        else:
            write_fcs(path, empty=True)
        with pytest.raises(ReaderFormatError, match="no cells"):
            _read_cell_table(ArtifactFile(path))

    @pytest.mark.parametrize("row", ["1,1,2,3", "1,1,2,3,4,5"])
    def test_malformed_row_width(self, tmp_path: Path, row: str) -> None:
        path = tmp_path / "cells.csv"
        path.write_text(f"cell_id,region,x,y,cyc001_ch001\n{row}\n", encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="row width"):
            _read_cell_table(ArtifactFile(path))

    def test_extra_zero_based_field_cannot_become_an_implicit_index(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n0,1,2,3,4,5\n1,1,2,3,4,5\n", encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="row width"):
            _read_cell_table(ArtifactFile(path))

    def test_explicit_empty_metadata_is_distinct_from_missing_field(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001,note\n1,1,0,0,1,\n", encoding="utf-8")
        result = _read_cell_table(ArtifactFile(path))
        assert result.obs["note"].isna().all()
        path.write_text("cell_id,region,x,y,cyc001_ch001,note\n1,1,0,0,1\n", encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="row width"):
            _read_cell_table(ArtifactFile(path))

    def test_bounded_header(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("x" * 1_048_577, encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="oversized"):
            _read_cell_table(ArtifactFile(path))

    def test_fcs_without_marker_labels(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.fcs"
        write_fcs(path, marker_labels=False)
        table = _read_cell_table(ArtifactFile(path))
        assert table.var.columns.tolist() == []
        assert table.obs["cell_id"].tolist() == [7, 2, 11]

    def test_string_region_and_uint64_identifiers(self, tmp_path: Path) -> None:
        frame = cell_frame()
        frame["region"] = "reg001"
        frame["cell_id"] = [str(2**64 - 1), "2", "3"]
        path = tmp_path / "cells.csv"
        write_csv(path, frame)
        result = codex(tmp_path)
        assert result.tables["table"].obs["cell_id"].dtype == np.dtype(np.uint64)
        assert result.shapes["reg001"].index.tolist() == [2**64 - 1, 2, 3]

    @pytest.mark.parametrize(
        "source",
        [
            "1,1,0,0,9007199254740993,0.5\n",
            "1,1,0,0,9007199254740993,0\n2,1,0,0,0.5,0\n",
        ],
    )
    def test_no_float_promotion_precision_loss(self, tmp_path: Path, source: str) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001,cyc002_ch001\n" + source, encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="precision"):
            _read_cell_table(ArtifactFile(path))

    def test_signed_unsigned_block_promotion(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n1,1,0,0,9223372036854775809\n2,1,0,0,-1\n", encoding="utf-8")
        mocker.patch("spatialdata_io.readers.codex._tables._CSV_BATCH_ROWS", 1)
        with pytest.raises(ReaderFormatError, match="precision"):
            _read_cell_table(ArtifactFile(path))

    def test_sparse_output_never_densifies(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_csv(tmp_path / "cells.csv")
        mocker.patch.object(sparse.csr_array, "toarray", side_effect=AssertionError("densification"))
        result = codex(tmp_path)
        assert isinstance(result.tables["table"].X, sparse.csr_array)
        assert result.tables["table"].X.nnz == 4

    def test_file_closed_after_schema_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n1,1,0,0,bad\n", encoding="utf-8")
        stream = path.open(encoding="utf-8")
        mocker.patch("pathlib.Path.open", return_value=stream)
        with pytest.raises(ReaderFormatError, match="nonnumeric"):
            _read_cell_table(ArtifactFile(path))
        assert stream.closed

    def test_owned_fcs_handle_closes_on_unexpected_parser_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "cells.fcs"
        write_fcs(path)
        stream = path.open("rb")
        mocker.patch("pathlib.Path.open", return_value=stream)
        mocker.patch(
            "spatialdata_io.readers.codex._tables.readfcs.ReadFCS", side_effect=RuntimeError("injected decoder failure")
        )
        with pytest.raises(RuntimeError, match="injected decoder failure") as caught:
            _read_cell_table(ArtifactFile(path))
        assert stream.closed
        assert any("FCS events" in note for note in caught.value.__notes__)

    def test_unrepresentable_large_decimal_is_rejected_before_float_inference(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n1,1,0,0,9007199254740993.0\n", encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="precision"):
            _read_cell_table(ArtifactFile(path))


class TestCellIds:
    """Preserve exact floating source integers at the FCS boundary."""

    def test_large_stored_float_is_not_shortened_decimal_string(self, tmp_path: Path) -> None:
        result = _cell_ids(pd.Series([float(2**63)]), ArtifactFile(tmp_path / "cells.fcs"))
        assert result.tolist() == [2**63]
        assert result.dtype == np.dtype(np.uint64)


class TestRegionName:
    """Reject unsupported scalar region types explicitly."""

    @pytest.mark.parametrize("value", [True, False, np.bool_(1), "", complex(1)])
    def test_invalid_region_scalar(self, tmp_path: Path, *, value: bool | str | complex) -> None:
        with pytest.raises(ReaderFormatError, match="region"):
            _region_name(pd.Series([value]), ArtifactFile(tmp_path / "cells.csv"))
