"""Curio owned H5AD and identifier-aligned enrichment."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.curio import curio
from spatialdata_io.readers.curio._layout import _discover_layout
from spatialdata_io.readers.curio._tables import _enrich_features, _enrich_metrics, _read_table
from tests.integration.readers.curio._fixtures import write_curio

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestReadTable:
    """Validate owned H5AD and eager enrichment schemas."""

    @pytest.mark.parametrize(
        "rows",
        ["bead-b\t1\t2\n", "bead-b\t\n", "bead-b\t1\nbead-b\t0\nbead-c\t1\n", "missing\t1\nbead-a\t0\nbead-c\t1\n"],
    )
    def test_invalid_clusters(self, tmp_path: Path, rows: str) -> None:
        write_curio(tmp_path)
        (tmp_path / "sample_cluster_assignment.txt").write_text(rows, encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="cluster"):
            _read_table(_discover_layout(tmp_path))

    @pytest.mark.parametrize("kind", ["empty", "duplicate_obs", "duplicate_var"])
    def test_invalid_h5ad_identifiers(self, tmp_path: Path, kind: str) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        table = ad.read_h5ad(path)
        if kind == "empty":
            table = table[:0].copy()
        elif kind == "duplicate_obs":
            table.obs_names = ["b", "b", "c"]
        else:
            table.var_names = ["b", "b", "c"]
        table.write_h5ad(path)

        with pytest.raises(ReaderFormatError, match="identifiers"):
            _read_table(_discover_layout(tmp_path))

    def test_missing_expression(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        table = ad.read_h5ad(path)
        table.X = None
        table.write_h5ad(path)

        with pytest.raises(ReaderFormatError, match="missing expression") as raised:
            _read_table(_discover_layout(tmp_path))

        assert raised.value.path == path
        assert raised.value.context.reader == "curio"

    def test_h5ad_failure_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_curio(tmp_path)
        error = OSError("injected read failure")
        mocker.patch("spatialdata_io.readers.curio._tables.ad.read_h5ad", side_effect=error)

        with pytest.raises(OSError, match="injected") as raised:
            _read_table(_discover_layout(tmp_path))

        assert any("curio" in note and "counts" in note for note in raised.value.__notes__)

    def test_sparse_expression_does_not_densify(self, tmp_path: Path, mocker: MockerFixture) -> None:
        write_curio(tmp_path)
        mocker.patch.object(sparse.csr_array, "toarray", side_effect=AssertionError("densification"))
        mocker.patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("densification"))

        assert isinstance(curio(tmp_path).tables["table"].X, sparse.csr_array)

    def test_nonfinite_expression(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_anndata.h5ad"
        table = ad.read_h5ad(path)
        table.X = sparse.csr_array([[np.inf, 0, 0], [0, 1, 0], [0, 0, 2]])
        table.write_h5ad(path)

        with pytest.raises(ValueError, match="finite"):
            curio(tmp_path)


class TestEnrichMetrics:
    """Validate metric names and category dictionaries."""

    @pytest.mark.parametrize(
        "rows",
        [
            "Name,Value\nReads,2\n",
            "Metric,Value,Category\nReads,1,QC\nReads,2,QC\n",
            "Metric,Value,Category\n,1,QC\n",
            "Metric,Value,Category\nReads,1,\n",
            "Metric,Value,Category\nReads,100,QC,extra\n",
            "Metric,Value,Category,Category\nReads,100,QC,extra\n",
        ],
    )
    def test_invalid_metrics(self, tmp_path: Path, rows: str) -> None:
        write_curio(tmp_path)
        (tmp_path / "sample_Metrics.csv").write_text(rows, encoding="utf-8")

        with pytest.raises(ReaderFormatError, match=r"metric|columns"):
            _enrich_metrics(ad.AnnData(), _discover_layout(tmp_path).metrics)

    def test_invalid_quoted_header_context(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_Metrics.csv"
        path.write_text('"Metric,Value,Category\n', encoding="utf-8")

        with pytest.raises(csv.Error) as raised:
            _enrich_metrics(ad.AnnData(), _discover_layout(tmp_path).metrics)

        assert any(str(path) in note for note in raised.value.__notes__)


class TestEnrichFeatures:
    """Join by explicit feature ID and retain count-column order."""

    @pytest.mark.parametrize(
        "rows",
        [
            "gene\ngene-b\n",
            "gene\tI\ngene-b\t0.5\ngene-b\t0.4\n",
            "gene\tI\nmissing\t0.5\n",
            "gene\tI\n\t0.5\n",
            "gene\tsymbol\ngene-b\t1\n",
            "gene\tI\ngene-b\tinf\n",
            "gene\tI\ngene-b\tNA\n",
            "gene\tI\ngene-b\t\n",
            "gene\tI\tI\ngene-b\t0.1\t0.2\n",
            "gene\tI\ngene-b\t0.1\t0.2\n",
            "gene\t\ngene-b\t0.1\n",
        ],
    )
    def test_malformed_moran(self, tmp_path: Path, rows: str) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_variable_features_spatial_moransi.txt"
        path.write_text(rows, encoding="utf-8")

        with pytest.raises(ReaderFormatError):
            _enrich_features(
                pd.DataFrame({"symbol": ["B"]}, index=["gene-b"]), _discover_layout(tmp_path).moran_features
            )

    def test_header_only_moran(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_variable_features_spatial_moransi.txt"
        path.write_text("gene\tI\n", encoding="utf-8")

        table = curio(tmp_path).tables["table"]

        assert table.var["I"].isna().all()
        assert table.var_names.tolist() == ["gene-b", "gene-a", "gene-c"]

    def test_boolean_statistics(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_variable_features_spatial_moransi.txt"
        path.write_text("gene\tvariable\ngene-b\tTRUE\ngene-a\tFALSE\ngene-c\tTRUE\n", encoding="utf-8")

        assert curio(tmp_path).tables["table"].var["variable"].tolist() == [True, False, True]

    def test_invalid_quoted_header_context(self, tmp_path: Path) -> None:
        write_curio(tmp_path)
        path = tmp_path / "sample_variable_features_spatial_moransi.txt"
        path.write_text('"gene\tI\n', encoding="utf-8")

        with pytest.raises(csv.Error) as raised:
            _enrich_features(pd.DataFrame(index=["gene-b"]), _discover_layout(tmp_path).moran_features)

        assert any(str(path) in note for note in raised.value.__notes__)
