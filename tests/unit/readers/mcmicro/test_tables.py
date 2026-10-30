"""MCQuant tables as one finite sparse table linked once to every measured label."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.mcmicro._layout import Quantification, Segmentation
from spatialdata_io.readers.mcmicro._schemas import STRING_DTYPE, MarkerPanel
from spatialdata_io.readers.mcmicro._tables import read_table

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
    label = Segmentation(name, "seg", "cell", ArtifactFile(tmp_path / f"{name}.ome.tif"))
    return Quantification(ArtifactFile(path), label)


def _dtypes(*quantifications: Quantification, dtype: type[np.integer] = np.uint32) -> dict[str, np.dtype[np.generic]]:
    return {table.target.key: np.dtype(dtype) for table in quantifications}


class TestReadTable:
    """Rows, features, IDs, annotations, and coordinates keep their exact source meaning."""

    def test_concatenation_sparsity_types_and_linkage(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mocker.patch("spatialdata_io.readers.mcmicro._tables._MAX_BATCH_ROWS", 1)
        first = _quantification(tmp_path, "s1", _HEADER + "7,0,1.25,1.5,2,4,x\n3,0.5,0,3.5,4,,\n")
        second = _quantification(tmp_path, "s2", "B,CellID,A,note,Area,Y_centroid,X_centroid\n0,7,2,y,1,6,5.5\n")
        table = read_table((first, second), _panel(), _dtypes(first, second))

        expression = table.X
        assert isinstance(expression, sparse.csr_array)
        assert expression.has_canonical_format
        assert expression.dtype == np.float64
        np.testing.assert_array_equal(expression.toarray(), [[0, 1.25], [0.5, 0], [2, 0]])
        assert table.obs_names.tolist() == ["0", "1", "2"]
        assert table.var_names.tolist() == ["A", "B"]
        assert table.var["marker_name"].tolist() == ["A", "B"]
        assert table.obs.columns.tolist() == ["region", "CellID", "Area", "note"]
        assert table.obs["region"].tolist() == ["s1_seg_cell", "s1_seg_cell", "s2_seg_cell"]
        assert table.obs["CellID"].tolist() == [7, 3, 7]
        assert table.obs["CellID"].dtype == np.dtype(np.int64)
        np.testing.assert_array_equal(table.obs["Area"].to_numpy(), [4, np.nan, 1])
        assert table.obs["note"].tolist() == ["x", "", "y"]
        np.testing.assert_array_equal(table.obsm["spatial"], [[1.5, 2], [3.5, 4], [5.5, 6]])
        assert get_table_keys(table) == (["s1_seg_cell", "s2_seg_cell"], "region", "CellID")

    def test_label_without_cells_is_not_a_region(self, tmp_path: Path) -> None:
        empty = _quantification(tmp_path, "s1", _HEADER)
        measured = _quantification(tmp_path, "s2", _HEADER + "1,1,2,0,0,1,a\n")
        table = read_table((empty, measured), _panel(), _dtypes(empty, measured))
        assert get_table_keys(table)[0] == "s2_seg_cell"

    def test_wide_marker_blocks_are_never_densified_as_a_whole(self, tmp_path: Path, mocker: MockerFixture) -> None:
        features = tuple(f"M{i}" for i in range(64))
        rows = [",".join(["1" if i == j else "0" for j in range(64)]) for i in range(64)]
        text = "CellID," + ",".join(features) + ",X_centroid,Y_centroid\n"
        text += "".join(f"{i + 1},{row},0,0\n" for i, row in enumerate(rows))
        table_source = _quantification(tmp_path, "s", text)
        panel = MarkerPanel(features, pd.DataFrame(index=pd.Index(features)))
        mocker.patch("spatialdata_io.readers.mcmicro._tables._BATCH_MEMORY_BYTES", 8 * 64 * 4)
        spy = mocker.spy(sparse, "vstack")
        table = read_table((table_source,), panel, _dtypes(table_source))
        assert spy.call_count == 1
        assert len(spy.call_args.args[0]) == 16
        assert isinstance(table.X, sparse.csr_array)
        assert table.X.nnz == 64

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            (_HEADER + "1,inf,0,0,0,1,a\n", "nonfinite"),
            (_HEADER + "1,1,0,0,0,1\n", "malformed quantification"),
            (_HEADER + "1,1,0,,0,1,a\n", "nonnumeric 'X_centroid'"),
            (_HEADER + "1,1,0,0,0,big,a\n", "nonnumeric 'Area'"),
            (_HEADER + "1,1,0,0,0,1,a\n1,2,0,0,0,1,b\n", "outside the label contract"),
            (_HEADER + "300,1,0,0,0,1,a\n", "outside the label contract"),
            ("CellID,A,B,X_centroid,Y_centroid,A\n", "unique nonempty"),
            ("CellID,A,X_centroid,Y_centroid\n1,1,0,0\n", "missing columns"),
            ('"unterminated\n', "malformed CSV header|unique nonempty"),
        ],
    )
    def test_rejects(self, tmp_path: Path, text: str, match: str) -> None:
        source = _quantification(tmp_path, "s", text)
        with pytest.raises(ReaderFormatError, match=match):
            read_table((source,), _panel(), _dtypes(source, dtype=np.uint8))

    def test_tables_share_one_schema(self, tmp_path: Path) -> None:
        first = _quantification(tmp_path, "s1", _HEADER + "1,1,0,0,0,1,a\n")
        second = _quantification(tmp_path, "s2", "CellID,A,B,X_centroid,Y_centroid\n1,1,0,0,0\n")
        with pytest.raises(ReaderFormatError, match="shared MCQuant schema"):
            read_table((first, second), _panel(), _dtypes(first, second))

    def test_no_quantified_cells_fail(self, tmp_path: Path) -> None:
        source = _quantification(tmp_path, "s", _HEADER)
        with pytest.raises(ReaderFormatError, match="no quantified cells"):
            read_table((source,), _panel(), _dtypes(source))
