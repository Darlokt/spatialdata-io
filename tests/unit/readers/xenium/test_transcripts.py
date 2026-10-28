"""Lazy transcript points: projection, partitions, known categories and per-task validation."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dask.callbacks import Callback
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity, get_transformation

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.xenium import _transcripts
from spatialdata_io.readers.xenium._layout import TabularSource
from spatialdata_io.readers.xenium._transcripts import read_transcripts
from tests.integration.readers.xenium import _fixtures as fx

if TYPE_CHECKING:
    from pathlib import Path

    import dask.dataframe as dd
    from pytest_mock import MockerFixture

COLUMNS_1_0 = ["transcript_id", "cell_id", "overlaps_nucleus", "feature_name", "x", "y", "z", "qv"]
COLUMNS_3_0 = [*COLUMNS_1_0, "fov_name", "nucleus_distance", "codeword_index", "codeword_category", "is_gene"]


def _cells(bundle: fx.XeniumFixture) -> pd.Index:
    if bundle.profile.family == "integer_ids":
        return pd.Index(np.asarray(bundle.cell_ids, dtype=np.int64))
    return pd.Index(bundle.cell_ids, dtype="str")


def _read(bundle: fx.XeniumFixture, kind: str = "parquet") -> dd.DataFrame:
    source = TabularSource(ArtifactFile(bundle.root / f"transcripts.{kind}"), "parquet" if kind == "parquet" else "csv")
    return read_transcripts(
        source, bundle.profile.family, features=pd.Index(bundle.features, dtype="str"), cells=_cells(bundle)
    )


def _expected_cell_ids(bundle: fx.XeniumFixture) -> list[object]:
    unassigned: object = -1 if bundle.profile.family == "integer_ids" else "UNASSIGNED"
    return [bundle.cell_ids[cell] if cell >= 0 else unassigned for cell in bundle.transcripts.cell_index]


def _rewrite(path: Path, column: str, values: pa.Array) -> None:
    """Replace one column and keep the row-group layout of the file."""
    rows = pq.read_metadata(path).row_group(0).num_rows
    table = pq.read_table(path)
    pq.write_table(table.set_column(table.column_names.index(column), column, values), path, row_group_size=rows)


class TestReadTranscripts:
    """Points stay lazy and reproduce every projected source row in file order."""

    @pytest.mark.parametrize(
        ("profile", "columns"),
        [
            (fx.XOA_1_0, COLUMNS_1_0),
            (fx.XOA_1_3, [*COLUMNS_1_0, "fov_name", "nucleus_distance"]),
            (fx.XOA_3_0, COLUMNS_3_0),
        ],
        ids=["integer-binary-names", "binary-ids", "polygon-sets"],
    )
    def test_values_types_and_laziness(self, tmp_path: Path, profile: fx.Profile, columns: list[str]) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, profile)
        tasks: list[object] = []

        with Callback(posttask=lambda *args: tasks.append(args)):
            points = _read(bundle)

        assert tasks == []
        assert list(points.columns) == columns
        assert points["feature_name"].cat.known
        assert points["feature_name"].cat.categories.tolist() == list(bundle.features)
        model_attrs = points.attrs["spatialdata_attrs"]  # pyrefly: ignore[unsupported-operation] - Dask types attrs as a method.
        assert model_attrs == {"feature_key": "feature_name", "instance_key": "cell_id"}
        assert isinstance(get_transformation(points, to_coordinate_system="global"), Identity)
        PointsModel.validate(points)

        frame = points.compute()
        truth = bundle.transcripts
        assert frame.index.equals(pd.RangeIndex(len(truth.x)))
        np.testing.assert_array_equal(frame["transcript_id"].to_numpy(), truth.transcript_id)
        np.testing.assert_array_equal(frame["x"].to_numpy(), truth.x)
        np.testing.assert_array_equal(frame["y"].to_numpy(), truth.y)
        np.testing.assert_array_equal(frame["z"].to_numpy(), truth.z)
        np.testing.assert_array_equal(frame["qv"].to_numpy(), truth.qv)
        np.testing.assert_array_equal(frame["overlaps_nucleus"].to_numpy(), truth.overlaps_nucleus)
        assert frame["x"].dtype == np.float32
        assert frame["transcript_id"].dtype == np.uint64
        assert frame["feature_name"].tolist() == [bundle.features[index] for index in truth.feature_index]
        assert frame["cell_id"].tolist() == _expected_cell_ids(bundle)
        expected_ids = np.int64 if profile.family == "integer_ids" else pd.ArrowDtype(pa.string())
        assert frame["cell_id"].dtype == expected_ids

    def test_row_group_partitions_have_global_divisions(self, tmp_path: Path, mocker: MockerFixture) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(transcript_row_group=7))
        whole = _read(fx.write_xenium_bundle(tmp_path / "whole", fx.XOA_3_0)).compute()

        # Tasks capture the function when the graph is built.
        reads = mocker.spy(_transcripts, "_read_row_group")
        points = _read(bundle)
        third = points.get_partition(2).compute()

        assert points.npartitions == 4
        assert points.divisions == (0, 7, 14, 21, 27)
        assert reads.call_count == 1
        pd.testing.assert_frame_equal(third, whole.iloc[14:21])
        pd.testing.assert_frame_equal(points.compute(), whole)

    # Blocks smaller than one record leave empty blocks between record boundaries.
    @pytest.mark.parametrize("block_bytes", [64, 256])
    def test_csv_blocks_equal_parquet(self, tmp_path: Path, mocker: MockerFixture, block_bytes: int) -> None:
        mocker.patch.object(_transcripts, "_BLOCK_BYTES", block_bytes)
        parquet = _read(fx.write_xenium_bundle(tmp_path / "parquet", fx.XOA_3_0)).compute()
        bundle = fx.write_xenium_bundle(tmp_path / "csv", fx.XOA_3_0, fx.Options(transcripts="csv"))

        points = _read(bundle, "csv")

        assert points.npartitions > 1
        assert not points.known_divisions
        frame = points.compute()
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), parquet)

    def test_projects_documented_columns_only(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        path = bundle.root / "transcripts.parquet"
        table = pq.read_table(path)
        pq.write_table(table.append_column("vendor_extra", pa.array(["x"] * table.num_rows)), path)

        assert list(_read(bundle).columns) == COLUMNS_3_0

    def test_empty_source_has_typed_columns(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        path = bundle.root / "transcripts.parquet"
        pq.write_table(pq.read_table(path).slice(0, 0), path)

        frame = _read(bundle).compute()

        assert frame.empty
        assert list(frame.columns) == COLUMNS_3_0
        assert frame["x"].dtype == np.float32

    @pytest.mark.parametrize(
        ("column", "values", "match"),
        [
            ("feature_name", ["NOT_IN_PANEL"], r"unknown feature_name values \['NOT_IN_PANEL'\]"),
            ("cell_id", ["zzzzzzzz-1"], r"unknown cell_id values \['zzzzzzzz-1'\]"),
            ("x_location", [np.inf], "nonfinite x_location values"),
            ("qv", [None], r"missing values in \['qv'\]"),
        ],
        ids=["feature", "cell", "nonfinite", "null"],
    )
    def test_invalid_rows_raise_when_computed(
        self, tmp_path: Path, column: str, values: list[object], match: str
    ) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(transcript_row_group=7))
        path = bundle.root / "transcripts.parquet"
        source = pq.read_table(path)[column]
        replaced = values + source.to_pylist()[len(values) :]
        _rewrite(path, column, pa.array(replaced, source.type))

        points = _read(bundle)

        # Only the partition holding the defect fails.
        points.get_partition(1).compute()
        with pytest.raises(ReaderFormatError, match=match):
            points.get_partition(0).compute()

    def test_integer_sentinel_needs_integer_ids(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_1_0)
        path = bundle.root / "transcripts.parquet"
        ids = pq.read_table(path)["cell_id"].to_pylist()
        _rewrite(path, "cell_id", pa.array([-2, *ids[1:]], pa.int32()))

        with pytest.raises(ReaderFormatError, match=r"unknown cell_id values \[-2\]"):
            _read(bundle).compute()

    def test_invalid_binary_feature_names(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_1_0)
        path = bundle.root / "transcripts.parquet"
        names = pq.read_table(path)["feature_name"].to_pylist()
        _rewrite(path, "feature_name", pa.array([b"\xff", *names[1:]], pa.binary()))

        with pytest.raises(ReaderFormatError, match="UTF-8"):
            _read(bundle).compute()

    def test_malformed_csv_records_raise(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(transcripts="csv"))
        path = bundle.root / "transcripts.csv"
        path.write_text(path.read_text() + "1,2\n")

        with pytest.raises(ReaderFormatError, match="malformed CSV"):
            _read(bundle, "csv").compute()

    def test_missing_documented_column(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0)
        path = bundle.root / "transcripts.parquet"
        pq.write_table(pq.read_table(path).drop_columns(["qv"]), path)

        with pytest.raises(ReaderFormatError, match="qv"):
            _read(bundle)

    def test_graph_holds_paths_not_handles(self, tmp_path: Path) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(transcript_row_group=7))
        points = _read(bundle)

        graph = dict(points.__dask_graph__())

        assert pickle.loads(pickle.dumps(graph)).keys() == graph.keys()  # noqa: S301, RUF100 - Test-owned graph.
