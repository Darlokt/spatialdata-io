"""Curio final normalization, failure context, and table linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import spatialdata_io.readers.curio._reader as curio_reader
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.curio import curio
from tests.integration.readers.curio._fixtures import write_curio

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestCurio:
    """Exercise final normalization and contextual parser failures."""

    @pytest.mark.parametrize(
        "suffix",
        [
            "cluster_assignment.txt",
            "Metrics.csv",
            "variable_features_clusters.txt",
            "variable_features_spatial_moransi.txt",
        ],
    )
    def test_empty_enrichment_context(self, tmp_path: Path, suffix: str) -> None:
        write_curio(tmp_path)
        path = tmp_path / f"sample_{suffix}"
        path.write_text("", encoding="utf-8")

        with pytest.raises((pd.errors.EmptyDataError, ReaderFormatError)) as raised:
            curio(tmp_path)

        details = str(raised.value) + " ".join(getattr(raised.value, "__notes__", []))

        assert "curio" in details
        assert str(path) in details

    def test_dense_expression_becomes_sparse(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        table = ad.read_h5ad(path)
        table.X = np.eye(3, dtype=np.float32)
        table.write_h5ad(path)

        result = curio(tmp_path).tables["table"]

        assert isinstance(result.X, sparse.csr_array)
        assert result.X.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(result.X.data, [1, 1, 1])

    def test_final_linkage_once_uses_materialized_shape_ids(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_curio(tmp_path)
        parse = mocker.spy(curio_reader, "parse_linked_table")

        result = curio(tmp_path)

        parse.assert_called_once()
        assert parse.call_args.kwargs["expected_instance_ids"]["cells"] is result.shapes["cells"].index
