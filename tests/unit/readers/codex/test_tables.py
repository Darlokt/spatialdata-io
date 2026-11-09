"""CODEX CSV and FCS cell-table parsing, sparse assembly and physical positions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from scipy import sparse

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.codex import _tables, codex
from spatialdata_io.readers.codex._tables import read_cell_table, set_positions
from tests.integration.readers.codex._fixtures import cell_frame, write_csv, write_fcs

if TYPE_CHECKING:
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

        result = read_cell_table(ArtifactFile(path)).table

        assert result.obs["note"].tolist() == [note] * 3

    @pytest.mark.parametrize(("note", "expected"), [('"a"suffix', "asuffix"), ('"unterminated', "unterminated\n")])
    def test_arrow_permissive_quoting(self, tmp_path: Path, note: str, expected: str) -> None:
        path = tmp_path / "cells.csv"
        path.write_text(f"cell_id,region,x,y,cyc001_ch001,note\n1,1,0,0,1,{note}\n", encoding="utf-8")
        assert read_cell_table(ArtifactFile(path)).table.obs["note"].tolist() == [expected]

    @pytest.mark.parametrize("spelling", ["+1", " 1 ", "0001", "-0"])
    def test_integral_expression_spellings(self, tmp_path: Path, spelling: str) -> None:
        """Retain accepted signed, padded, and whitespace-surrounded integer tokens."""
        path = tmp_path / "cells.csv"
        path.write_text(f"cell_id,region,x,y,cyc001_ch001\n1,1,0,0,{spelling}\n", encoding="utf-8")

        result = read_cell_table(ArtifactFile(path)).table

        assert isinstance(result.X, sparse.csr_array)
        assert result.X.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(result.X.data, [] if spelling == "-0" else [1])

    @pytest.mark.parametrize("spelling", [" 1.5 ", "1.5\t"])
    def test_padded_decimal_spellings(self, tmp_path: Path, spelling: str) -> None:
        """Accept whitespace around decimals as around integers."""
        path = tmp_path / "cells.csv"
        path.write_text(f"cell_id,region,x,y,cyc001_ch001\n1,1,{spelling},0,{spelling}\n", encoding="utf-8")

        cells = read_cell_table(ArtifactFile(path))

        assert isinstance(cells.table.X, sparse.csr_array)
        np.testing.assert_array_equal(cells.table.X.data, [1.5])
        np.testing.assert_array_equal(cells.coordinates, [[1.5, 0]])

    def test_padded_underflow_follows_arrow_rounding(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n1,1,0,0, 1e-400 \n", encoding="utf-8")
        result = read_cell_table(ArtifactFile(path)).table
        assert isinstance(result.X, sparse.csr_array)
        assert result.X.nnz == 0
        assert result.X.dtype == np.dtype(np.float64)

    @pytest.mark.parametrize("column", ["cell_id", "region", "x", "y", "note", "cyc001_ch001"])
    def test_nul_rejected_in_every_source_field(self, tmp_path: Path, column: str) -> None:
        """Reject NULs rather than allowing the CSV decoder to truncate source values."""
        frame = cell_frame()
        frame[column] = ["1\x00suffix", "2", "3"]
        path = tmp_path / "cells.csv"
        write_csv(path, frame)

        with pytest.raises(ReaderFormatError, match=r"NUL|malformed CSV"):
            read_cell_table(ArtifactFile(path))

    @pytest.mark.parametrize("dtype", [np.int16, np.uint32, np.float32, np.float64])
    def test_fcs_native_numeric_widths(self, tmp_path: Path, mocker: MockerFixture, dtype: type[np.number]) -> None:
        """Keep native FCS signal widths while normalizing observation annotations."""
        path = tmp_path / "cells.fcs"
        path.touch()
        frame = cell_frame().abs().astype(dtype) if np.dtype(dtype).kind == "u" else cell_frame().astype(dtype)
        source = mocker.Mock(data=frame, channels=pd.DataFrame({"PnN": frame.columns}))
        mocker.patch("spatialdata_io.readers.codex._tables.readfcs.ReadFCS", return_value=source)

        result = read_cell_table(ArtifactFile(path)).table

        assert isinstance(result.X, sparse.csr_array)
        assert result.X.dtype == np.dtype(dtype)
        expected = frame[["cyc002_ch001", "cyc001_ch002"]].to_numpy().ravel()
        np.testing.assert_array_equal(result.X.data, expected[expected != 0])

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("cyc002_ch001", "bad", "nonnumeric"),
            ("x", 2**53 + 1, "precision loss"),
            ("x", float("nan"), "invalid x/y"),
        ],
    )
    def test_invalid_native_fcs_fields(
        self, tmp_path: Path, mocker: MockerFixture, field: str, value: str | float, match: str
    ) -> None:
        path = tmp_path / "cells.fcs"
        path.touch()
        frame = cell_frame()
        frame[field] = pd.Series([value] * len(frame), dtype="int64" if isinstance(value, int) else None)
        source = mocker.Mock(data=frame, channels=pd.DataFrame({"PnN": frame.columns}))
        mocker.patch("spatialdata_io.readers.codex._tables.readfcs.ReadFCS", return_value=source)
        with pytest.raises(ReaderFormatError, match=match):
            read_cell_table(ArtifactFile(path))

    def test_native_fcs_mixed_expression_cannot_round_integers(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "cells.fcs"
        path.touch()
        frame = cell_frame()
        frame["cyc002_ch001"] = pd.Series([2**53 + 1] * len(frame), dtype="int64")
        source = mocker.Mock(data=frame, channels=pd.DataFrame({"PnN": frame.columns}))
        mocker.patch("spatialdata_io.readers.codex._tables.readfcs.ReadFCS", return_value=source)
        with pytest.raises(ReaderFormatError, match="precision loss"):
            read_cell_table(ArtifactFile(path))

    def test_inconsistent_fcs_parameter_count_fails(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "cells.fcs"
        path.touch()
        frame = cell_frame()
        source = mocker.Mock(data=frame.iloc[:, :-1], channels=pd.DataFrame({"PnN": frame.columns}))
        mocker.patch("spatialdata_io.readers.codex._tables.readfcs.ReadFCS", return_value=source)
        with pytest.raises(ReaderFormatError, match="inconsistent FCS parameters"):
            read_cell_table(ArtifactFile(path))

    def test_csv_source_blocks_preserve_all_rows(self, tmp_path: Path, mocker: MockerFixture) -> None:
        rows = 20_001
        path = tmp_path / "cells.csv"
        path.write_text(
            "cell_id,region,x,y,cyc001_ch001,cyc001_ch002\n"
            + "".join(f"{row + 1},1,{row},2,{row % 3},{-(row % 5)}\n" for row in range(rows)),
            encoding="utf-8",
        )
        whole = read_cell_table(ArtifactFile(path)).table
        mocker.patch.object(_tables, "_CSV_BLOCK_BYTES", 4096)
        blocks = mocker.spy(_tables, "_csv_expression")
        small = read_cell_table(ArtifactFile(path)).table
        assert len(blocks.call_args_list) > 1
        assert sum(call.args[0].num_rows for call in blocks.call_args_list) == rows
        expected, actual = whole.X, small.X
        assert isinstance(expected, sparse.csr_array)
        assert isinstance(actual, sparse.csr_array)
        np.testing.assert_array_equal(expected.data, actual.data)
        np.testing.assert_array_equal(expected.indices, actual.indices)
        np.testing.assert_array_equal(expected.indptr, actual.indptr)
        assert isinstance(whole.obs, pd.DataFrame)
        assert isinstance(small.obs, pd.DataFrame)
        pd.testing.assert_frame_equal(whole.obs, small.obs, check_exact=True)

    def test_wide_csv_expression_stays_in_source_batches(self, tmp_path: Path, mocker: MockerFixture) -> None:
        rows, width = 101, 128
        columns = [f"cyc{i:03d}_ch001" for i in range(width)]
        values = np.zeros((rows, width), dtype=np.float64)
        values[np.arange(rows), np.arange(rows) % width] = -1.25
        frame = pd.DataFrame(values, columns=columns)
        frame["cell_id"], frame["region"], frame["x"], frame["y"] = np.arange(rows), "001", 1.5, -2.5
        write_csv(tmp_path / "cells.csv", frame)
        mocker.patch.object(_tables, "_CSV_BLOCK_BYTES", 8192)
        blocks = mocker.spy(_tables, "_csv_expression")
        result = read_cell_table(ArtifactFile(tmp_path / "cells.csv")).table
        assert len(blocks.call_args_list) > 1
        assert all(call.args[0].num_rows < rows for call in blocks.call_args_list)
        assert isinstance(result.X, sparse.csr_array)
        np.testing.assert_array_equal(result.X.data, np.full(rows, -1.25))
        np.testing.assert_array_equal(result.X.indices, np.arange(rows))
        assert result.X.dtype == np.dtype(np.float64)

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
        ],
    )
    def test_invalid_source_values(self, tmp_path: Path, column: str, values: list[str | int | float]) -> None:
        """Reject invalid source intensities, identifiers, coordinates, and region relationships."""
        frame = cell_frame()
        frame[column] = values
        path = tmp_path / "cells.csv"
        write_csv(path, frame)

        with pytest.raises(
            (ReaderFormatError, ValueError),
            match=r"finite|cell_id|regions|region|numeric|precision|overflow|x/y|malformed CSV",
        ):
            read_cell_table(ArtifactFile(path))

    @pytest.mark.parametrize("column", ["x", "y", "cyc002_ch001"])
    @pytest.mark.parametrize("value", ["1e-400", "-1e-400"])
    def test_required_scientific_fields_follow_arrow_underflow(self, tmp_path: Path, column: str, value: str) -> None:
        frame = cell_frame()
        frame[column] = value
        write_csv(tmp_path / "cells.csv", frame)
        cells = read_cell_table(ArtifactFile(tmp_path / "cells.csv"))
        if column == "cyc002_ch001":
            assert isinstance(cells.table.X, sparse.csr_array)
            assert cells.table.X[:, 0].nnz == 0
        else:
            np.testing.assert_array_equal(cells.coordinates[:, 0 if column == "x" else 1], 0)

    @pytest.mark.parametrize("source_format", ["csv", "fcs"])
    def test_empty_cells(self, tmp_path: Path, source_format: str) -> None:
        """Reject an empty population for both supported table containers."""
        path = tmp_path / f"cells.{source_format}"
        if source_format == "csv":
            write_csv(path, cell_frame().iloc[:0])
        else:
            write_fcs(path, empty=True)

        with pytest.raises(ReaderFormatError, match="no cells"):
            read_cell_table(ArtifactFile(path))

    @pytest.mark.parametrize("body", ["", "\n", "\n\r\n"])
    def test_header_and_blank_lines_have_no_cells(self, tmp_path: Path, body: str) -> None:
        """Blank lines after the header are not records."""
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n" + body, encoding="utf-8", newline="")

        with pytest.raises(ReaderFormatError, match="no cells"):
            read_cell_table(ArtifactFile(path))

    def test_blank_lines_before_records_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n\n5,1,0,0,2\n\n", encoding="utf-8")

        cells = read_cell_table(ArtifactFile(path))

        assert cells.table.n_obs == 1
        assert cells.cell_ids.tolist() == [5]

    def test_invalid_utf8_record(self, tmp_path: Path) -> None:
        """Undecodable record bytes are a malformed table, not a bare decoding error."""
        path = tmp_path / "cells.csv"
        path.write_bytes(b"cell_id,region,x,y,cyc001_ch001,note\n1,1,0,0,1,\xff\n")

        with pytest.raises(ReaderFormatError, match=r"artifact 'cell table'.*UTF.?8"):
            read_cell_table(ArtifactFile(path))

    def test_open_failure_has_reader_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "cells.csv"
        write_csv(path)
        mocker.patch("spatialdata_io.readers.codex._tables.csv_header", return_value=list(cell_frame().columns))
        mocker.patch("pathlib.Path.open", side_effect=PermissionError("injected"))

        with pytest.raises(PermissionError, match="injected") as caught:
            read_cell_table(ArtifactFile(path))

        assert any("cell table" in note for note in caught.value.__notes__)

    def test_validated_population(self, tmp_path: Path) -> None:
        """Return final IDs, native coordinates and the region name with the table."""
        path = tmp_path / "cells.csv"
        write_csv(path)

        cells = read_cell_table(ArtifactFile(path))

        assert cells.region == "1"
        assert cells.cell_ids.dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(cells.cell_ids, [7, 2, 11])
        np.testing.assert_array_equal(cells.coordinates, [[20, 1.5], [2.5, 30], [13, 8]])
        assert cells.table.obs["cell_id"].tolist() == [7, 2, 11]
        assert cells.table.obs["region"].tolist() == ["1"] * 3

    @pytest.mark.parametrize("row", ["1,1,2,3", "1,1,2,3,4,5"])
    def test_malformed_row_width(self, tmp_path: Path, row: str) -> None:
        """Require exactly one physical field per declared source column."""
        path = tmp_path / "cells.csv"
        path.write_text(f"cell_id,region,x,y,cyc001_ch001\n{row}\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            read_cell_table(ArtifactFile(path))

    def test_extra_zero_based_field_cannot_become_an_implicit_index(self, tmp_path: Path) -> None:
        """Prevent surplus fields from silently shifting cell metadata into an inferred index."""
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n0,1,2,3,4,5\n1,1,2,3,4,5\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            read_cell_table(ArtifactFile(path))

    def test_explicit_empty_metadata_is_distinct_from_missing_field(self, tmp_path: Path) -> None:
        """Distinguish an explicit empty optional field from a physically missing field."""
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001,note\n1,1,0,0,1,\n", encoding="utf-8")

        result = read_cell_table(ArtifactFile(path)).table

        assert result.obs["note"].tolist() == [""]
        assert not result.obs["note"].isna().any()

        path.write_text("cell_id,region,x,y,cyc001_ch001,note\n1,1,0,0,1\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            read_cell_table(ArtifactFile(path))

    def test_fcs_without_marker_labels(self, tmp_path: Path) -> None:
        """Keep cycle feature IDs valid when optional FCS marker labels are absent."""
        path = tmp_path / "cells.fcs"
        write_fcs(path, marker_labels=False)

        table = read_cell_table(ArtifactFile(path)).table

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

    @pytest.mark.parametrize("token", ["9007199254740993", "9007199254740993.0", "1e23"])
    def test_float64_expression_rounds_directly(self, tmp_path: Path, token: str) -> None:
        path = tmp_path / "cells.csv"
        path.write_text(f"cell_id,region,x,y,cyc001_ch001\n1,1,{token},0,{token}\n", encoding="utf-8")
        cells = read_cell_table(ArtifactFile(path))
        assert isinstance(cells.table.X, sparse.csr_array)
        assert cells.table.X.dtype == np.dtype(np.float64)
        assert cells.table.X.data.tolist() == [float(token)]
        assert cells.coordinates[0, 0] == float(token)

    def test_float64_rounding_and_signed_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv"
        path.write_text(
            "cell_id,region,x,y,cyc001_ch001\n0,001,12.947243945462901,-0.0,0.10000000000000002\n",
            encoding="utf-8",
        )
        cells = read_cell_table(ArtifactFile(path))
        assert cells.coordinates[0, 0].hex() == "0x1.9e4fd288e31c0p+3"
        assert np.signbit(cells.coordinates[0, 1])
        assert isinstance(cells.table.X, sparse.csr_array)
        assert cells.table.X.data.tolist() == [0.10000000000000002]
        assert cells.cell_ids.dtype == np.dtype(np.uint64)

    @pytest.mark.parametrize("column", ["cell_id", "x", "y", "cyc001_ch001"])
    def test_empty_required_numeric_field_fails(self, tmp_path: Path, column: str) -> None:
        values = {"cell_id": "0", "region": "001", "x": "1", "y": "2", "cyc001_ch001": "-3"}
        values[column] = ""
        path = tmp_path / "cells.csv"
        path.write_text(",".join(values) + "\n" + ",".join(values.values()) + "\n", encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            read_cell_table(ArtifactFile(path))

    def test_literal_region_spellings_cannot_collapse(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n1,001,0,0,1\n2,1,0,0,1\n", encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="multiple regions"):
            read_cell_table(ArtifactFile(path))

    def test_mixed_large_signed_expression_uses_fixed_float64(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n1,1,0,0,9223372036854775809\n2,1,0,0,-1\n", encoding="utf-8")
        mocker.patch.object(_tables, "_CSV_BLOCK_BYTES", 64)
        result = read_cell_table(ArtifactFile(path)).table
        assert isinstance(result.X, sparse.csr_array)
        assert result.X.data.tolist() == [float(2**63 + 1), -1.0]

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
        opened = mocker.spy(Path, "open")

        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            read_cell_table(ArtifactFile(path))

        assert opened.spy_return_list
        assert all(stream.closed for stream in opened.spy_return_list)

    @pytest.mark.parametrize("failure", ["none", "parse", "finite", "annotation"])
    def test_csv_streams_close_on_all_boundaries(self, tmp_path: Path, mocker: MockerFixture, failure: str) -> None:
        path = tmp_path / "cells.csv"
        value = "bad" if failure == "parse" else "nan" if failure == "finite" else "1"
        note = "bad\x00text" if failure == "annotation" else "NA"
        path.write_text(f"cell_id,region,x,y,cyc001_ch001,note\n0,001,1,2,{value},{note}\n", encoding="utf-8")
        opened = mocker.spy(Path, "open")
        if failure == "none":
            read_cell_table(ArtifactFile(path))
        else:
            with pytest.raises(ValueError, match=r"finite|malformed CSV|NUL"):
                read_cell_table(ArtifactFile(path))
        assert opened.spy_return_list
        assert all(stream.closed for stream in opened.spy_return_list)

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
            read_cell_table(ArtifactFile(path))

        assert stream.closed
        assert any("cell table" in note for note in caught.value.__notes__)

    @pytest.mark.parametrize("token", ["0x10", " 16 ", "\t16\t"])
    def test_unsigned_id_grammar(self, tmp_path: Path, token: str) -> None:
        path = tmp_path / "cells.csv"
        path.write_text(f"cell_id,region,x,y,cyc001_ch001\n{token},001,0,0,-1\n", encoding="utf-8")
        result = read_cell_table(ArtifactFile(path))
        assert result.cell_ids.tolist() == [16]
        assert result.region == "001"

    @pytest.mark.parametrize("token", ["-1", "-0", "+1", "2.0", "2e0", "1_0", "", str(2**64)])
    def test_invalid_unsigned_id_spellings(self, tmp_path: Path, token: str) -> None:
        path = tmp_path / "cells.csv"
        path.write_text(f"cell_id,region,x,y,cyc001_ch001\n{token},1,0,0,1\n", encoding="utf-8")
        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            read_cell_table(ArtifactFile(path))

    @pytest.mark.parametrize("batch_bytes", [256, 1024 * 1024])
    def test_exact_signed_unsigned_expression_is_batch_independent(
        self, tmp_path: Path, mocker: MockerFixture, batch_bytes: int
    ) -> None:
        """Compatible signed and unsigned intensities may share exact float64 storage."""
        path = tmp_path / "cells.csv"
        path.write_text("cell_id,region,x,y,cyc001_ch001\n1,1,0,0,-1\n2,1,0,0,9223372036854775808\n", encoding="utf-8")
        mocker.patch("spatialdata_io.readers.codex._tables._CSV_BLOCK_BYTES", batch_bytes)

        result = read_cell_table(ArtifactFile(path)).table

        assert isinstance(result.X, sparse.csr_array)
        np.testing.assert_array_equal(result.X.data, [-1.0, float(2**63)])

    @pytest.mark.parametrize("batch_bytes", [256, 1024 * 1024])
    @pytest.mark.parametrize("all_zero", [False, True])
    def test_single_and_multiple_blocks_preserve_canonical_sparse_values(
        self, tmp_path: Path, mocker: MockerFixture, batch_bytes: int, *, all_zero: bool
    ) -> None:
        """Retain CSR shape, values, dtype and row order even when all signals are zero."""
        frame = cell_frame()
        frame[["cyc002_ch001", "cyc001_ch002"]] = 0 if all_zero else [[0, 2], [-3, 0], [4, 5]]
        write_csv(tmp_path / "cells.csv", frame)
        mocker.patch("spatialdata_io.readers.codex._tables._CSV_BLOCK_BYTES", batch_bytes)

        result = read_cell_table(ArtifactFile(tmp_path / "cells.csv")).table

        assert isinstance(result.X, sparse.csr_array)
        assert not isinstance(result.X, sparse.spmatrix)
        assert result.X.has_canonical_format
        assert result.X.shape == (3, 2)
        assert result.X.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(result.X.data, [] if all_zero else [2, -3, 4, 5])
        np.testing.assert_array_equal(result.X.indices, [] if all_zero else [1, 0, 0, 1])
        np.testing.assert_array_equal(result.X.indptr, [0, 0, 0, 0] if all_zero else [0, 1, 2, 4])
        assert result.obs["cell_id"].tolist() == [7, 2, 11]

    @pytest.mark.parametrize("batch_bytes", [256, 1024 * 1024])
    @pytest.mark.parametrize("region", ["001", "1.0", "-1", "0"])
    def test_literal_regions_and_fixed_coordinates(
        self, tmp_path: Path, mocker: MockerFixture, batch_bytes: int, region: str
    ) -> None:
        path = tmp_path / "cells.csv"
        path.write_text(
            f"cell_id,region,x,y,cyc001_ch001\n7,{region},1,-2,0\n2,{region},2.5,0,-3\n11,{region},-4,3.5,4\n",
            encoding="utf-8",
        )
        mocker.patch.object(_tables, "_CSV_BLOCK_BYTES", batch_bytes)
        cells = read_cell_table(ArtifactFile(path))
        assert cells.region == region
        assert cells.table.obs["region"].tolist() == [region] * 3
        assert cells.table.obs["x"].dtype == pd.ArrowDtype(pa.float64())
        assert cells.table.obs["y"].dtype == pd.ArrowDtype(pa.float64())
        np.testing.assert_array_equal(cells.coordinates, [[1, -2], [2.5, 0], [-4, 3.5]])


class TestSetPositions:
    """Physical finalization borrows native XY and preserves owned expression."""

    def test_expression_and_native_buffers_are_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "cells.csv"
        write_csv(path)
        cells = read_cell_table(ArtifactFile(path))
        table, coordinates = cells.table, cells.coordinates
        expression = table.X
        original = coordinates.copy()
        set_positions(table, coordinates, pixel_size_um=0.5, calibration="Caller-declared source metric.")
        assert table.X is expression
        np.testing.assert_array_equal(coordinates, original)
        assert not np.shares_memory(table.obsm["spatial"], coordinates)

    @pytest.mark.parametrize(
        ("coordinate", "scale"),
        [
            (1e-100, 1e-300),
            (-1e-100, 1e-300),
            (0.25, float.fromhex("0x0.0000000000001p-1022")),
            (-0.25, float.fromhex("0x0.0000000000001p-1022")),
            (2.0, 1e308),
        ],
    )
    def test_invalid_conversion_precedes_mutation(self, tmp_path: Path, coordinate: float, scale: float) -> None:
        """Reject zero loss and overflow before replacing positions or their metadata."""
        path = tmp_path / "cells.csv"
        write_csv(path)
        table = read_cell_table(ArtifactFile(path)).table
        coordinates = np.full((3, 2), coordinate, dtype=np.float64)
        original = coordinates.copy()
        expression = table.X
        previous_positions = np.arange(6, dtype=np.float64).reshape(3, 2)
        previous_metadata = {"sentinel": "existing position metadata"}
        table.obsm["spatial"] = previous_positions
        table.uns[POSITIONS_KEY] = previous_metadata

        with np.errstate(over="ignore", under="ignore"), pytest.raises(ValueError, match="physical"):
            set_positions(table, coordinates, pixel_size_um=scale, calibration="Caller metric.")

        assert table.obsm["spatial"] is previous_positions
        assert table.uns[POSITIONS_KEY] is previous_metadata
        assert table.X is expression
        np.testing.assert_array_equal(coordinates, original)

    @pytest.mark.parametrize(
        ("coordinate", "scale", "expected"),
        [
            (0.0, 1e-300, 0.0),
            (-0.0, float.fromhex("0x0.0000000000001p-1022"), -0.0),
            (float.fromhex("0x1p-1022"), 0.5, float.fromhex("0x0.8p-1022")),
            (-float.fromhex("0x1p-1022"), 0.5, -float.fromhex("0x0.8p-1022")),
            (
                float.fromhex("0x0.0000000000001p-1022"),
                0.75,
                float.fromhex("0x0.0000000000001p-1022"),
            ),
            (
                -float.fromhex("0x0.0000000000001p-1022"),
                0.75,
                -float.fromhex("0x0.0000000000001p-1022"),
            ),
            (-3.0, 0.5, -1.5),
        ],
    )
    def test_zero_subnormal_and_negative_positions(
        self, tmp_path: Path, coordinate: float, scale: float, expected: float
    ) -> None:
        """Keep true zero, representable subnormals, and ordinary negative coordinates."""
        path = tmp_path / "cells.csv"
        write_csv(path)
        table = read_cell_table(ArtifactFile(path)).table
        coordinates = np.full((3, 2), coordinate, dtype=np.float64)
        original = coordinates.copy()

        with np.errstate(under="ignore"):
            set_positions(table, coordinates, pixel_size_um=scale, calibration="Caller metric.")

        np.testing.assert_array_equal(table.obsm["spatial"], np.full((3, 2), expected))
        physical = table.obsm["spatial"]
        assert isinstance(physical, np.ndarray)
        np.testing.assert_array_equal(np.signbit(physical), np.signbit(np.full((3, 2), expected)))
        np.testing.assert_array_equal(coordinates, original)
        assert table.obsm["spatial"].dtype == np.dtype(np.float64)
