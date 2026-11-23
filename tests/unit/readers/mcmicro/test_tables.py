"""MCQuant tables as one finite sparse table linked once to every measured label."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.mcmicro import _tables as mcmicro_tables
from spatialdata_io.readers.mcmicro._layout import Quantification, Segmentation
from spatialdata_io.readers.mcmicro._rasters import ImageGrid
from spatialdata_io.readers.mcmicro._schemas import STRING_DTYPE, MarkerPanel, quantification_columns
from spatialdata_io.readers.mcmicro._tables import _read_measurements, read_table

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_HEADER = "CellID,A,B,X_centroid,Y_centroid,Area,note\n"


def _panel() -> MarkerPanel:
    """Return a fresh reader-owned panel, as read_markers does for every call."""
    index = pd.Index(["A", "B"])
    names = pd.Series(["A", "B"], index=index, dtype=STRING_DTYPE)
    return MarkerPanel(("A", "B"), pd.DataFrame({"marker_name": names}, index=index))


def _quantification(tmp_path: Path, name: str, text: str) -> Quantification:
    path = tmp_path / f"{name}.csv"
    path.write_text(text, encoding="utf-8")
    label = Segmentation(name, "seg", "cell", ArtifactFile(tmp_path / f"{name}.ome.tif"), f"{name}_seg_cell")
    return Quantification(ArtifactFile(path), label)


def _dtypes(*quantifications: Quantification, dtype: type[np.integer] = np.uint32) -> dict[str, np.dtype[np.integer]]:
    return {table.target.key: np.dtype(dtype) for table in quantifications}


def _grids(
    *quantifications: Quantification, sizes: tuple[float | None, float | None] = (None, None)
) -> dict[str, ImageGrid]:
    return {
        table.target.image_id: ImageGrid(table.target.image_id, (10, 10), sizes, table.target.image_id)
        for table in quantifications
    }


def _grid(sizes: tuple[float | None, float | None], name: str = "image", frame: str = "global") -> ImageGrid:
    return ImageGrid(name, (10, 10), sizes, frame)


class TestReadTable:
    """Rows, features, IDs, annotations, and coordinates keep their exact source meaning."""

    def test_concatenation_sparsity_types_and_linkage(self, tmp_path: Path) -> None:
        first = _quantification(tmp_path, "s1", _HEADER + "7,0,1.25,1.5,2,4,x\n3,0.5,0,3.5,4,,\n")
        second = _quantification(tmp_path, "s2", "B,CellID,A,note,Area,Y_centroid,X_centroid\n0,7,2,y,1,6,5.5\n")
        table = read_table((first, second), _panel(), _dtypes(first, second), _grids(first, second))

        expression = table.X
        assert isinstance(expression, sparse.csr_array)
        assert expression.has_canonical_format
        assert expression.dtype == np.float64
        np.testing.assert_array_equal(expression.toarray(), [[0, 1.25], [0.5, 0], [2, 0]])
        assert table.obs_names.tolist() == ["0", "1", "2"]
        assert table.var_names.tolist() == ["A", "B"]
        assert table.var["marker_name"].tolist() == ["A", "B"]
        assert table.obs.columns.tolist() == ["region", "CellID", "Area", "note", "spatial_pixel_frame"]
        assert table.obs["region"].tolist() == ["s1_seg_cell", "s1_seg_cell", "s2_seg_cell"]
        assert table.obs["CellID"].tolist() == [7, 3, 7]
        assert table.obs["CellID"].dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(table.obs["Area"].to_numpy(), [4, np.nan, 1])
        assert table.obs["note"].tolist() == ["x", "", "y"]
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[1.5, 2], [3.5, 4], [5.5, 6]])
        assert table.obs["spatial_pixel_frame"].tolist() == ["s1_regionprops_indices"] * 2 + ["s2_regionprops_indices"]
        assert "spatial" not in table.obsm
        assert get_table_keys(table) == (["s1_seg_cell", "s2_seg_cell"], "region", "CellID")

    def test_source_columns_yield_to_reader_owned_names(self, tmp_path: Path) -> None:
        """Generated and linkage columns keep their names; case variants from the header are renamed."""
        header = "CellID,A,B,X_centroid,Y_centroid,Region,cellid,Spatial_Frame\n"
        first = _quantification(tmp_path, "s1", header + "7,0,1,1.5,2,r,c,f\n")
        second = _quantification(tmp_path, "s2", header + "3,1,0,3.5,4,r,c,f\n")
        grids = {"s1": _grid((None, None), "s1", "s1"), "s2": _grid((None, None), "s2", "s2")}

        with pytest.warns(UserWarning, match="quantification CSV header") as caught:
            table = read_table((first, second), _panel(), _dtypes(first, second), grids)

        assert len(caught) == 1
        assert "'Region' -> 'Region_1', 'cellid' -> 'cellid_1', 'Spatial_Frame' -> 'Spatial_Frame_1'" in str(
            caught[0].message
        )
        assert table.obs.columns.tolist() == [
            "region",
            "CellID",
            "Region_1",
            "cellid_1",
            "Spatial_Frame_1",
            "spatial_pixel_frame",
        ]
        assert get_table_keys(table) == (["s1_seg_cell", "s2_seg_cell"], "region", "CellID")

    def test_label_without_cells_is_not_a_region(self, tmp_path: Path) -> None:
        empty = _quantification(tmp_path, "s1", _HEADER)
        measured = _quantification(tmp_path, "s2", _HEADER + "1,1,2,0,0,1,a\n")
        table = read_table((empty, measured), _panel(), _dtypes(empty, measured), _grids(empty, measured))
        assert get_table_keys(table)[0] == "s2_seg_cell"

    def test_empty_first_table_defines_annotation_order(self, tmp_path: Path) -> None:
        empty = _quantification(tmp_path, "s1", "CellID,A,B,X_centroid,Y_centroid,Area,noteA,noteB\n")
        measured = _quantification(
            tmp_path, "s2", "noteB,B,Area,noteA,CellID,Y_centroid,A,X_centroid\nv,2,3,u,1,5,4,6\n"
        )
        table = read_table((empty, measured), _panel(), _dtypes(empty, measured), _grids(empty, measured))
        assert table.obs.columns.tolist() == ["region", "CellID", "Area", "noteA", "noteB"]
        assert table.obs["noteA"].tolist() == ["u"]
        assert table.obs["noteB"].tolist() == ["v"]
        np.testing.assert_array_equal(table.obsm["spatial_pixels"], [[6, 5]])

    def test_wide_marker_blocks_are_never_densified_as_a_whole(self, tmp_path: Path, mocker: MockerFixture) -> None:
        features = tuple(f"M{i}" for i in range(64))
        rows = [",".join(["1" if i == j else "0" for j in range(64)]) for i in range(64)]
        text = "CellID," + ",".join(features) + ",X_centroid,Y_centroid\n"
        text += "".join(f"{i + 1},{row},0,0\n" for i, row in enumerate(rows))
        table_source = _quantification(tmp_path, "s", text)
        panel = MarkerPanel(features, pd.DataFrame(index=pd.Index(features)))
        mocker.patch.object(mcmicro_tables, "_BATCH_BYTES", 1024)
        blocks = mocker.spy(mcmicro_tables, "as_csr_array")
        table = read_table((table_source,), panel, _dtypes(table_source), _grids(table_source))
        assert blocks.call_count > 1
        assert all(call.args[0].nbytes <= 4 * 1024 for call in blocks.call_args_list)
        assert all(call.args[0].shape[1] == len(features) for call in blocks.call_args_list)
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.nnz == 64
        np.testing.assert_array_equal(table.X.data, np.ones(64))
        np.testing.assert_array_equal(table.X.indices, np.arange(64))
        np.testing.assert_array_equal(table.X.indptr, np.arange(65))

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            (_HEADER + "1,inf,0,0,0,1,a\n", "nonfinite"),
            (_HEADER + "1,nan,0,0,0,1,a\n", "nonfinite"),
            (_HEADER + "1,-inf,0,0,0,1,a\n", "nonfinite"),
            (_HEADER + "1,1,0,0,0,1\n", r"malformed CSV \("),
            (_HEADER + "1,1,0,,0,1,a\n", "missing 'X_centroid'"),
            (_HEADER + "300,1,0,0,0,1,a\n", "'CellID' value 300 above 255"),
            (_HEADER + "2.0,1,0,0,0,1,a\n", r"malformed CSV \("),
            ("CellID,A,B,X_centroid,Y_centroid,A\n", "repeated column"),
            ("CellID,A,X_centroid,Y_centroid\n1,1,0,0\n", "missing columns"),
        ],
    )
    def test_rejects(self, tmp_path: Path, text: str, match: str) -> None:
        source = _quantification(tmp_path, "s", text)
        with pytest.raises(ReaderFormatError, match=match) as caught:
            read_table((source,), _panel(), _dtypes(source, dtype=np.uint8), _grids(source))
        assert caught.value.context == ReaderErrorContext("mcmicro", "quantification")
        assert caught.value.path == source.path

    def test_duplicate_ids_across_batches_fail_at_final_linkage(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch.object(mcmicro_tables, "_BATCH_BYTES", 128)
        text = _HEADER + "".join(f"{identifier},1,0,0,0,1,note\n" for identifier in [*range(1, 31), 7])
        source = _quantification(tmp_path, "s", text)
        with pytest.raises(ReaderFormatError, match=r"duplicate \(region, instance\) pairs") as caught:
            read_table((source,), _panel(), _dtypes(source), _grids(source))
        assert caught.value.context.reader == "mcmicro"
        assert caught.value.context.artifact == "cell table"
        assert "s_seg_cell" in caught.value.found
        assert "7" in caught.value.found

    def test_tables_share_one_schema(self, tmp_path: Path) -> None:
        first = _quantification(tmp_path, "s1", _HEADER + "1,1,0,0,0,1,a\n")
        second = _quantification(tmp_path, "s2", "CellID,A,B,X_centroid,Y_centroid\n1,1,0,0,0\n")
        with pytest.raises(ReaderFormatError, match="shared MCQuant schema"):
            read_table((first, second), _panel(), _dtypes(first, second), _grids(first, second))

    def test_no_quantified_cells_fail(self, tmp_path: Path) -> None:
        source = _quantification(tmp_path, "s", _HEADER)
        with pytest.raises(ReaderFormatError, match="no quantified cells"):
            read_table((source,), _panel(), _dtypes(source), _grids(source))

    def test_uint64_cell_ids(self, tmp_path: Path) -> None:
        source = _quantification(tmp_path, "s", _HEADER + f"{2**64 - 1},1,0,0,0,1,a\n")
        table = read_table((source,), _panel(), _dtypes(source, dtype=np.uint64), _grids(source))
        assert table.obs["CellID"].tolist() == [2**64 - 1]
        assert table.obs["CellID"].dtype == np.dtype(np.uint64)

    def test_required_numeric_empty_fields_fail(self, tmp_path: Path) -> None:
        source = _quantification(tmp_path, "s", _HEADER + ",1,0,0,0,1,\n")
        with pytest.raises(ReaderFormatError, match="missing 'CellID'"):
            read_table((source,), _panel(), _dtypes(source), _grids(source))

    @pytest.mark.parametrize("token", ["NA", "nan"])
    def test_optional_morphology_only_empty_is_missing(self, tmp_path: Path, token: str) -> None:
        source = _quantification(tmp_path, "s", _HEADER + f"1,1,0,0,0,{token},a\n")
        with pytest.raises(ReaderFormatError, match=r"malformed CSV|nonfinite"):
            read_table((source,), _panel(), _dtypes(source), _grids(source))

    def test_oversized_record_fails_with_source_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch.object(mcmicro_tables, "_BATCH_BYTES", 128)
        source = _quantification(tmp_path, "s", _HEADER + "1,1,0,0,0,1," + "x" * 1024 + "\n")
        with pytest.raises(ReaderFormatError, match="malformed CSV") as caught:
            read_table((source,), _panel(), _dtypes(source), _grids(source))
        assert caught.value.path == source.path


class TestReadMeasurements:
    """The reader owns and closes its source on every iteration outcome."""

    @pytest.mark.parametrize("failure", [None, "numeric", "parser"])
    def test_source_closes(self, tmp_path: Path, mocker: MockerFixture, failure: str | None) -> None:
        text = _HEADER + "".join(f"{i},1,0,0,0,1,a\n" for i in range(1, 21))
        if failure == "numeric":
            text += "2,inf,0,0,0,1,a\n"
        elif failure == "parser":
            text += "2,1,0,0,0,1\n"
        source = io.BytesIO(text.encode())
        source_table = _quantification(tmp_path, "s", _HEADER)
        path = source_table.path
        columns = quantification_columns(_HEADER.strip().split(","), ("A", "B"), path)
        mocker.patch("pathlib.Path.open", return_value=source)
        mocker.patch.object(mcmicro_tables, "_BATCH_BYTES", 128)
        if failure is None:
            assert (
                _read_measurements(source_table, _grid((None, None)), columns, np.dtype(np.uint32)).annotations.num_rows
                == 20
            )
        else:
            with pytest.raises(ReaderFormatError, match=r"nonfinite|malformed CSV") as caught:
                _read_measurements(source_table, _grid((None, None)), columns, np.dtype(np.uint32))
            assert caught.value.context.reader == "mcmicro"
            assert caught.value.path == path
        assert source.closed

    def test_open_error_preserves_type_and_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        source_table = _quantification(tmp_path, "s", _HEADER)
        path = source_table.path
        columns = quantification_columns(_HEADER.strip().split(","), ("A", "B"), path)
        mocker.patch("pathlib.Path.open", side_effect=PermissionError("denied"))
        with pytest.raises(PermissionError) as caught:
            _read_measurements(source_table, _grid((None, None)), columns, np.dtype(np.uint32))
        assert any("mcmicro" in note and str(path) in note for note in caught.value.__notes__)


class TestCalibratedMeasurements:
    def test_failure_identifies_the_unrepresentable_core(self, tmp_path: Path) -> None:
        first = _quantification(tmp_path, "a", _HEADER + "1,1,2,2,1,4,x\n")
        second = _quantification(tmp_path, "b", _HEADER + "1,1,2,2,1,4,x\n")
        grids = {"a": _grid((2, 3), "core_1", "core_1"), "b": _grid((1e308, 3), "core_2", "core_2")}
        with pytest.raises(ReaderFormatError, match="nonfinite physical positions in 'core_2'") as caught:
            read_table((first, second), _panel(), _dtypes(first, second), grids)
        assert caught.value.path == second.path
        assert caught.value.context == ReaderErrorContext("mcmicro", "quantification")
