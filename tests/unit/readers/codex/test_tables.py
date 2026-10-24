"""CODEX layout, source schemas, identifier precision, and image options."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.codex import _tables, codex
from spatialdata_io.readers.codex._tables import _read_cell_table
from tests.integration.readers.codex._fixtures import cell_frame, write_csv, write_fcs

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestReadCellTable:
    """Exercise real source parser failures before geometry and linkage."""

    @pytest.mark.parametrize("note", ["quoted, field", 'embedded "quote"', "two\nlines", "NA", "NULL"])
    def test_quoted_and_literal_metadata(self, tmp_path: Path, note: str) -> None:
        """Preserve quoted delimiters, embedded newlines, and literal null-like text."""
        frame = cell_frame()
        frame["note"] = note
        path = tmp_path / "cells.csv"
        write_csv(path, frame)

        result = _read_cell_table(ArtifactFile(path))

        assert result.obs["note"].tolist() == [note] * 3

    @pytest.mark.parametrize("note", ['"a"suffix', '"unterminated'])
    def test_malformed_quoting_is_rejected(self, tmp_path: Path, note: str) -> None:
        """Reject malformed quoting even when a permissive decoder could recover field values."""
        path = tmp_path / "cells.csv"
        path.write_text(f"cell_id,region,x,y,cyc001_ch001,note\n1,1,0,0,1,{note}\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="quoting"):
            _read_cell_table(ArtifactFile(path))

    @pytest.mark.parametrize("spelling", ["+1", " 1 ", "0001", "-0"])
    def test_integral_expression_spellings(self, tmp_path: Path, spelling: str) -> None:
        """Retain accepted signed, padded, and whitespace-surrounded integer tokens."""
        path = tmp_path / "cells.csv"
        path.write_text(f"cell_id,region,x,y,cyc001_ch001\n1,1,0,0,{spelling}\n", encoding="utf-8")

        result = _read_cell_table(ArtifactFile(path))

        assert isinstance(result.X, sparse.csr_array)
        assert result.X.dtype == np.dtype(np.int64)
        np.testing.assert_array_equal(result.X.data, [] if spelling == "-0" else [1])

    @pytest.mark.parametrize("column", ["cell_id", "note", "cyc001_ch001"])
    def test_nul_rejected_in_every_source_field(self, tmp_path: Path, column: str) -> None:
        """Reject NULs rather than allowing the CSV decoder to truncate source values."""
        frame = cell_frame()
        frame[column] = ["1\x00suffix", "2", "3"]
        path = tmp_path / "cells.csv"
        write_csv(path, frame)

        with pytest.raises(ReaderFormatError, match="NUL"):
            _read_cell_table(ArtifactFile(path))

    @pytest.mark.parametrize("dtype", [np.int16, np.uint32, np.float32, np.float64])
    def test_fcs_native_numeric_widths(self, tmp_path: Path, mocker: MockerFixture, dtype: type[np.number]) -> None:
        """Keep native FCS signal widths while normalizing observation annotations."""
        path = tmp_path / "cells.fcs"
        path.touch()
        frame = cell_frame().abs().astype(dtype) if np.dtype(dtype).kind == "u" else cell_frame().astype(dtype)
        source = mocker.Mock(data=frame, channels=pd.DataFrame({"PnN": frame.columns}))
        mocker.patch("spatialdata_io.readers.codex._tables.readfcs.ReadFCS", return_value=source)

        result = _read_cell_table(ArtifactFile(path))

        assert isinstance(result.X, sparse.csr_array)
        assert result.X.dtype == np.dtype(dtype)
        expected = frame[["cyc002_ch001", "cyc001_ch002"]].to_numpy().ravel()
        np.testing.assert_array_equal(result.X.data, expected[expected != 0])

    def test_dense_csv_batches_are_bounded_by_feature_width(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Cap each numerical materialization by bytes while preserving final sparse buffers."""
        frame = cell_frame()
        path = tmp_path / "cells.csv"
        write_csv(path, frame)
        whole = _read_cell_table(ArtifactFile(path))
        mocker.patch("spatialdata_io.readers.codex._tables._CSV_MAX_DENSE_BYTES", 2 * 8)
        split = mocker.spy(_tables, "_split_cells")

        bounded = _read_cell_table(ArtifactFile(path))

        assert split.call_count == len(frame)
        assert all(call.args[0].shape[0] == 1 for call in split.call_args_list)
        assert isinstance(whole.X, sparse.csr_array)
        assert isinstance(bounded.X, sparse.csr_array)
        np.testing.assert_array_equal(bounded.X.data, whole.X.data)
        np.testing.assert_array_equal(bounded.X.indices, whole.X.indices)
        np.testing.assert_array_equal(bounded.X.indptr, whole.X.indptr)
        assert isinstance(whole.obs, pd.DataFrame)
        assert isinstance(bounded.obs, pd.DataFrame)
        pd.testing.assert_frame_equal(bounded.obs, whole.obs, check_exact=True)

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
        """Reject invalid source intensities, identifiers, coordinates, and region relationships."""
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
        """Reject an empty population for both supported table containers."""
        path = tmp_path / f"cells.{source_format}"
        if source_format == "csv":
            write_csv(path, cell_frame().iloc[:0])
        else:
            write_fcs(path, empty=True)

        with pytest.raises(ReaderFormatError, match="no cells"):
            _read_cell_table(ArtifactFile(path))

    @pytest.mark.parametrize("row", ["1,1,2,3", "1,1,2,3,4,5"])
    def test_malformed_row_width(self, tmp_path: Path, row: str) -> None:
        """Require exactly one physical field per declared source column."""
        path = tmp_path / "cells.csv"
        path.write_text(f"cell_id,region,x,y,cyc001_ch001\n{row}\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="row width"):
            _read_cell_table(ArtifactFile(path))

    def test_extra_zero_based_field_cannot_become_an_implicit_index(self, tmp_path: Path) -> None:
        """Prevent surplus fields from silently shifting cell metadata into an inferred index."""
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n0,1,2,3,4,5\n1,1,2,3,4,5\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="row width"):
            _read_cell_table(ArtifactFile(path))

    def test_explicit_empty_metadata_is_distinct_from_missing_field(self, tmp_path: Path) -> None:
        """Distinguish an explicit empty optional field from a physically missing field."""
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001,note\n1,1,0,0,1,\n", encoding="utf-8")

        result = _read_cell_table(ArtifactFile(path))

        assert result.obs["note"].isna().all()

        path.write_text("cell_id,region,x,y,cyc001_ch001,note\n1,1,0,0,1\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="row width"):
            _read_cell_table(ArtifactFile(path))

    def test_bounded_header(self, tmp_path: Path) -> None:
        """Reject an oversized CSV header before parsing the event population."""
        path = tmp_path / "cells.csv"
        path.write_text("x" * 1_048_577, encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="oversized"):
            _read_cell_table(ArtifactFile(path))

    def test_fcs_without_marker_labels(self, tmp_path: Path) -> None:
        """Keep cycle feature IDs valid when optional FCS marker labels are absent."""
        path = tmp_path / "cells.fcs"
        write_fcs(path, marker_labels=False)

        table = _read_cell_table(ArtifactFile(path))

        assert table.var.columns.tolist() == []
        assert table.obs["cell_id"].tolist() == [7, 2, 11]

    def test_string_region_and_uint64_identifiers(self, tmp_path: Path) -> None:
        """Link exact unsigned cell IDs within a named source region."""
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
        """Reject intensities changed by within-block numeric promotion."""
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001,cyc002_ch001\n" + source, encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="precision"):
            _read_cell_table(ArtifactFile(path))

    def test_signed_unsigned_block_promotion(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Reject inexact signed and unsigned intensity promotion across CSV blocks."""
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n1,1,0,0,9223372036854775809\n2,1,0,0,-1\n", encoding="utf-8")
        mocker.patch("spatialdata_io.readers.codex._tables._CSV_BATCH_ROWS", 1)

        with pytest.raises(ReaderFormatError, match="precision"):
            _read_cell_table(ArtifactFile(path))

    def test_sparse_output_never_densifies(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Keep expression sparse through geometry, compatibility conversion, and linkage."""
        write_csv(tmp_path / "cells.csv")
        mocker.patch.object(sparse.csr_array, "toarray", side_effect=AssertionError("densification"))

        result = codex(tmp_path)

        assert isinstance(result.tables["table"].X, sparse.csr_array)
        assert result.tables["table"].X.nnz == 4

    def test_file_closed_after_schema_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Close the owned CSV stream after a source-schema failure."""
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n1,1,0,0,bad\n", encoding="utf-8")
        stream = path.open(encoding="utf-8")
        mocker.patch("pathlib.Path.open", return_value=stream)

        with pytest.raises(ReaderFormatError, match="nonnumeric"):
            _read_cell_table(ArtifactFile(path))

        assert stream.closed

    def test_owned_fcs_handle_closes_on_unexpected_parser_failure(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Close the owned binary stream even after an unexpected FCS decoder failure."""
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
        """Reject a decimal token whose integral value would round during float parsing."""
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n1,1,0,0,9007199254740993.0\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="precision"):
            _read_cell_table(ArtifactFile(path))

    @pytest.mark.parametrize("batch_rows", [1, 10_000])
    def test_exact_signed_unsigned_expression_is_batch_independent(
        self, tmp_path: Path, mocker: MockerFixture, batch_rows: int
    ) -> None:
        """Compatible signed and unsigned intensities may share exact float64 storage."""
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n1,1,0,0,-1\n2,1,0,0,9223372036854775808\n", encoding="utf-8")
        mocker.patch("spatialdata_io.readers.codex._tables._CSV_BATCH_ROWS", batch_rows)

        result = _read_cell_table(ArtifactFile(path))

        assert isinstance(result.X, sparse.csr_array)
        np.testing.assert_array_equal(result.X.data, [-1.0, float(2**63)])
