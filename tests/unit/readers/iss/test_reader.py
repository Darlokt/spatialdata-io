"""ISS orchestration options and final sparse/linkage boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile
from dask.callbacks import Callback
from scipy import sparse
from spatialdata.models import TableModel

import spatialdata_io.readers._utils.image.tiff._dask as tiff_dask
import spatialdata_io.readers.iss._reader as iss_reader
from spatialdata_io.readers.iss import iss
from tests.integration.readers.iss.conftest import write_iss_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestIss:
    """Fail invalid public configuration before filesystem access."""

    @pytest.mark.parametrize("dataset_id", ["", "a/b", "a\\b", "__private", "has space"])
    def test_invalid_dataset_id(self, tmp_path: Path, dataset_id: str) -> None:
        with pytest.raises(ValueError, match="dataset_id"):
            iss(
                tmp_path / "absent",
                raw_image_file="raw.tiff",
                labels_file="labels.tiff",
                table_file="cells.h5ad",
                dataset_id=dataset_id,
            )

    @pytest.mark.parametrize("instance_key", ["", "region"])
    def test_invalid_instance_key(self, tmp_path: Path, instance_key: str) -> None:
        with pytest.raises(ValueError, match="instance_key"):
            iss(
                tmp_path / "absent",
                raw_image_file="raw.tiff",
                labels_file="labels.tiff",
                table_file="cells.h5ad",
                instance_key=instance_key,
            )

    def test_pyramids_construct_without_tasks_and_link_once(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_iss_dataset(tmp_path)
        source_file = tmp_path / "cells.h5ad"
        table = ad.read_h5ad(source_file)
        table.layers["metadata"] = np.zeros((2, 3), dtype=np.uint8)
        table.write_h5ad(source_file)
        parse = mocker.spy(iss_reader, "parse_linked_table")
        pixels = mocker.spy(tiff_dask, "_read_tiff_selection")
        tasks = mocker.Mock()
        with Callback(posttask=tasks):
            result = iss(
                tmp_path,
                raw_image_file="raw.tiff",
                labels_file="labels.tiff",
                table_file="cells.h5ad",
                image_scale_factors=[2, 2],
                labels_scale_factors=[2, 2],
            )
        tasks.assert_not_called()
        pixels.assert_not_called()
        parse.assert_called_once()
        assert "expected_instance_ids" not in parse.call_args.kwargs
        assert parse.call_args.kwargs["expression_layers"] == ("counts",)
        assert isinstance(result.tables["table"].layers["metadata"], np.ndarray)
        assert "region" not in ad.read_h5ad(source_file).obs

    @pytest.mark.parametrize("failure", ["duplicates", "empty", "nonfinite", "counts_nonfinite", "stale"])
    def test_table_failure_preserves_context_and_never_reads_labels(
        self, tmp_path: Path, mocker: MockerFixture, failure: str
    ) -> None:
        write_iss_dataset(tmp_path)
        table_file = tmp_path / "cells.h5ad"
        table = ad.read_h5ad(table_file)
        assert isinstance(table.obs, pd.DataFrame)
        if failure == "duplicates":
            table.obs["segmentation_id"] = [7, 7]
        elif failure == "empty":
            table = table[:0].copy()
        elif failure == "stale":
            table.uns[TableModel.ATTRS_KEY] = {"region": "old"}
        elif failure == "counts_nonfinite":
            table.layers["counts"] = np.full((2, 3), np.nan)
        else:
            table.X = np.full((2, 3), np.nan)
        table.write_h5ad(table_file)
        pixels = mocker.spy(tiff_dask, "_read_tiff_selection")
        with pytest.raises(ValueError, match=r"duplicate|regions|finite|absent") as caught:
            iss(
                tmp_path,
                raw_image_file="raw.tiff",
                labels_file="labels.tiff",
                table_file="cells.h5ad",
                instance_key="segmentation_id",
            )
        pixels.assert_not_called()
        assert any("iss" in note and str(table_file) in note for note in caught.value.__notes__)

    def test_sparse_expression_is_never_densified(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_iss_dataset(tmp_path)
        rows = np.arange(1000)
        expression = sparse.coo_array((np.ones(1000, dtype=np.float32), (rows, rows)), shape=(1000, 20_000)).tocsr()
        table = ad.AnnData(X=expression, obs=pd.DataFrame(index=[str(value) for value in range(1, 1001)]))
        table.write_h5ad(tmp_path / "cells.h5ad")
        labels = np.zeros(32 * 40, dtype=np.uint16)
        labels[:1000] = np.arange(1, 1001, dtype=np.uint16)
        tifffile.imwrite(tmp_path / "labels.tiff", labels.reshape(32, 40), photometric="minisblack")
        for storage in (sparse.csr_array, sparse.csr_matrix):
            mocker.patch.object(storage, "toarray", side_effect=AssertionError("expression densified"))
        result = iss(tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad")
        counts = result.tables["table"].X
        assert isinstance(counts, sparse.csr_array)
        assert counts.shape == (1000, 20_000)
        assert counts.nnz == 1000
        np.testing.assert_array_equal(counts.data, np.ones(1000, dtype=np.float32))
