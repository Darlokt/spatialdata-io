"""Opt-in secondary analysis aligned to cells by exact barcode."""

from __future__ import annotations

import io
import tarfile
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile
from spatialdata_io.readers.xenium import _analysis
from spatialdata_io.readers.xenium._analysis import read_analysis
from spatialdata_io.readers.xenium._layout import AnalysisArchive, AnalysisDirectory
from tests.integration.readers.xenium import _fixtures as fx

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

BARCODES = pd.Index(["a-1", "b-1", "c-1"], dtype="str")
CLUSTERS = "clustering/graphclust/clusters.csv"
PCA = "pca/gene_expression_2_components/projection.csv"


def _directory(root: Path, members: dict[str, str]) -> AnalysisDirectory:
    for name, text in members.items():
        (root / "analysis" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "analysis" / name).write_text(text)
    return AnalysisDirectory(ArtifactDirectory(root / "analysis"))


class TestReadAnalysis:
    """Clusterings and projections follow the given barcodes; absent cells are missing."""

    @pytest.mark.parametrize("kind", ["directory", "archive"])
    def test_vendor_layouts(self, tmp_path: Path, kind: Literal["directory", "archive"]) -> None:
        bundle = fx.write_xenium_bundle(tmp_path, fx.XOA_3_0, fx.Options(analysis=kind))
        source = (
            AnalysisDirectory(ArtifactDirectory(tmp_path / "analysis"))
            if kind == "directory"
            else AnalysisArchive(ArtifactFile(tmp_path / "analysis.tar.gz"))
        )

        analysis = read_analysis(source, pd.Index(bundle.barcodes, dtype="str"))

        clusters = analysis.clusters["gene_expression_graphclust"]
        assert clusters.categories.tolist() == ["1", "2"]
        assert clusters.categories.dtype == "str"
        assert clusters.codes.tolist() == [0, 1, 0, -1]
        assert sorted(analysis.projections) == ["pca_gene_expression_3_components", "umap_gene_expression_2_components"]
        pca = analysis.projections["pca_gene_expression_3_components"]
        assert pca.dtype == np.float64
        np.testing.assert_array_equal(pca[:3], [[0.5, -0.25, 0.125], [1.5, -1.25, 0.125], [2.5, -2.25, 0.125]])
        assert np.isnan(pca[3]).all()
        np.testing.assert_array_equal(
            analysis.projections["umap_gene_expression_2_components"][:3], [[0.75, 0.0], [1.75, 1.0], [2.75, 2.0]]
        )

    def test_clusters_follow_numeric_order_and_rows_any_order(self, tmp_path: Path) -> None:
        source = _directory(tmp_path, {CLUSTERS: "Barcode,Cluster\nc-1,10\na-1,2\nb-1,1\n"})

        clusters = read_analysis(source, BARCODES).clusters["graphclust"]

        assert clusters.categories.tolist() == ["1", "2", "10"]
        assert list(clusters) == ["2", "1", "10"]

    def test_unrelated_members_are_ignored(self, tmp_path: Path) -> None:
        source = _directory(
            tmp_path,
            {
                "diffexp/graphclust/differential_expression.csv": "Feature ID\n",
                "pca/gene_expression_2_components/components.csv": "PC,a\n",
                "clustering/graphclust/other.csv": "x\n",
            },
        )

        analysis = read_analysis(source, BARCODES)

        assert analysis.clusters == {}
        assert analysis.projections == {}

    @pytest.mark.parametrize(
        ("members", "match"),
        [
            ({CLUSTERS: "Barcode,Cluster,Extra\na-1,1,0\n"}, "Barcode,Cluster"),
            ({CLUSTERS: "Barcode,Cluster\nz-1,1\n"}, "unknown or repeated barcodes"),
            ({CLUSTERS: "Barcode,Cluster\na-1,1\na-1,2\n"}, "unknown or repeated barcodes"),
            ({CLUSTERS: "Barcode,Cluster\na-1,0\n"}, "nonpositive clusters"),
            ({CLUSTERS: "Barcode,Cluster\na-1,x\n"}, "malformed CSV"),
            ({PCA: "Barcode,PC-1,PC-3\na-1,1,2\n"}, r"Barcode,PC-1,\.\.\."),
            ({PCA: "Barcode\na-1\n"}, r"Barcode,PC-1,\.\.\."),
            ({PCA: "Barcode,PC-1,PC-2\na-1,inf,2\n"}, "nonfinite projections"),
            ({PCA: "Barcode,PC-1,PC-2\na-1,,2\n"}, "nonfinite projections"),
            ({PCA: '"Barcode\n'}, "malformed header"),
        ],
        ids=[
            "cluster-header",
            "unknown",
            "repeated",
            "nonpositive",
            "text-cluster",
            "components",
            "no-components",
            "infinite",
            "empty",
            "header",
        ],
    )
    def test_rejects_malformed_members(self, tmp_path: Path, members: dict[str, str], match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match):
            read_analysis(_directory(tmp_path, members), BARCODES)

    def test_archive_members_are_bounded(self, tmp_path: Path, mocker: MockerFixture) -> None:
        payload = b"Barcode,Cluster\na-1,1\n"
        with tarfile.open(tmp_path / "analysis.tar.gz", "w:gz") as archive:
            info = tarfile.TarInfo(f"analysis/{CLUSTERS}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        mocker.patch.object(_analysis, "MAX_MEMBER_BYTES", len(payload) - 1)

        with pytest.raises(ReaderFormatError, match="a member within 2 GiB"):
            read_analysis(AnalysisArchive(ArtifactFile(tmp_path / "analysis.tar.gz")), BARCODES)
