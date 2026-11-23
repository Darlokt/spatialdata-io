"""Shared count ingestion, HD population validation and independent linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd import _tables as tables

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestReadCounts:
    """HD requires populated axes and a unique barcode per observation."""

    @pytest.mark.parametrize("barcodes", [["007", "NA"], ["NA", "007"]])
    def test_keeps_literal_order_and_owned_buffers(
        self, tmp_path: Path, mocker: MockerFixture, barcodes: list[str]
    ) -> None:
        counts = csr_array([[1], [3]], dtype=np.float32)
        obs = pd.DataFrame(index=pd.Index(barcodes))
        var = pd.DataFrame(index=["Gene"])
        mocker.patch.object(tables, "read_10x_h5_components", return_value=(counts, obs, var))

        result, rows, features = tables.read_counts(ArtifactFile(tmp_path / "counts.h5"), version="4.1.0")

        assert result is counts
        assert rows is obs
        assert features is var
        assert rows.index.tolist() == barcodes

    @pytest.mark.parametrize(
        ("names", "features", "reason"),
        [
            (["same", "same"], ["Gene"], "duplicate barcodes"),
            ([], ["Gene"], "positive observation and feature dimensions"),
            (["spot"], [], "positive observation and feature dimensions"),
        ],
    )
    def test_rejects_invalid_population(
        self, tmp_path: Path, mocker: MockerFixture, names: list[str], features: list[str], reason: str
    ) -> None:
        path = ArtifactFile(tmp_path / "counts.h5")
        components = csr_array((len(names), len(features))), pd.DataFrame(index=names), pd.DataFrame(index=features)
        mocker.patch.object(tables, "read_10x_h5_components", return_value=components)
        with pytest.raises(ReaderFormatError, match=reason) as error:
            tables.read_counts(path, version="4.1.0")
        assert error.value.path == path


class TestLinkTable:
    """Link only against the independent materialized target IDs."""

    def test_linkage_and_serializable_categories(self, tmp_path: Path) -> None:
        result = ad.AnnData(csr_array([[1], [2]]), obs=pd.DataFrame(index=["first", "second"]))
        result.obs["instance_id"] = pd.Series([2, 5], index=result.obs_names, dtype="int64[pyarrow]")
        result = tables.link_table(
            result,
            "boundaries",
            pd.Index([2, 5]),
            position="vendor cell centroid",
            counts_path=ArtifactFile(tmp_path / "matrix.h5"),
            version="4.1.0",
        )
        assert result.uns["spatialdata_attrs"]["region"] == "boundaries"
        assert result.obs["region"].cat.categories.tolist() == ["boundaries"]
        result.write_h5ad(tmp_path / "roundtrip.h5ad")

    def test_membership(self, tmp_path: Path) -> None:
        result = ad.AnnData(csr_array([[1], [2]]), obs=pd.DataFrame(index=["first", "second"]))
        result.obs["instance_id"] = [2, 5]
        with pytest.raises(ReaderFormatError):
            tables.link_table(
                result,
                "boundaries",
                pd.Index([2, 3]),
                position="vendor cell centroid",
                counts_path=ArtifactFile(tmp_path / "matrix.h5"),
                version="4.1.0",
            )
