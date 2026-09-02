"""Tests for Visium position parsing, sparse alignment, and linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile, DatasetRoot
from spatialdata_io.readers.visium._layout import H5Counts, VisiumLayout
from spatialdata_io.readers.visium._metadata import CountMetadata, ScaleFactors
from spatialdata_io.readers.visium._tables import TableOptions, link_table, prepare_table, read_positions

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _layout(root: Path, positions: Path, *, headered: bool) -> VisiumLayout:
    counts = root / "counts.h5"
    counts.touch(exist_ok=True)
    scales = root / "scales.json"
    scales.touch(exist_ok=True)
    return VisiumLayout(
        root=DatasetRoot(ArtifactDirectory(root)),
        counts=H5Counts(ArtifactFile(counts), None),
        positions=ArtifactFile(positions),
        position_format="headered" if headered else "headerless",
        scale_factors=ArtifactFile(scales),
        hires_image=None,
        lowres_image=None,
        fullres_image=None,
    )


def _position_rows() -> str:
    return "b,0,1,2,30.5,40.5\na,1,0,3,10,20\nextra,1,2,1,50,60\n"


def _options() -> TableOptions:
    return TableOptions(
        dataset_id="sample",
        metadata=CountMetadata("sample", "chemistry", "spaceranger-2.1.0"),
        scale_factors=ScaleFactors(0.5, 0.25, 6.0, {"spot_diameter_fullres": 6.0}),
        make_var_names_unique=True,
    )


class TestReadPositions:
    """Validate both vendor CSV schemas and all coordinate columns."""

    @pytest.mark.parametrize("headered", [False, True])
    def test_reads_schema_without_reordering_rows(self, tmp_path: Path, *, headered: bool) -> None:
        path = tmp_path / "positions.csv"
        header = "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n" if headered else ""
        path.write_text(header + _position_rows(), encoding="utf-8")

        result = read_positions(_layout(tmp_path, path, headered=headered))

        assert result.index.tolist() == ["b", "a", "extra"]
        assert result["in_tissue"].dtype == np.dtype(np.int64)
        assert result["array_row"].tolist() == [1, 0, 2]
        np.testing.assert_array_equal(result["pxl_col_in_fullres"], np.asarray([40.5, 20.0, 60.0]))

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            ("a,2,0,0,1,2\n", "binary in_tissue"),
            ("a,1,-1,0,1,2\n", "nonnegative integer"),
            ("a,1,0.5,0,1,2\n", "integer values"),
            ("a,1,0,0,nan,2\n", "finite numeric"),
            ("a,1,0,0,-1,2\n", "nonnegative pixel"),
            ("a,1,0,0,1,2\na,0,1,1,3,4\n", "duplicate barcodes"),
        ],
    )
    def test_rejects_invalid_external_values(self, tmp_path: Path, row: str, message: str) -> None:
        path = tmp_path / "positions.csv"
        path.write_text(row, encoding="utf-8")

        with pytest.raises(ReaderFormatError, match=message):
            read_positions(_layout(tmp_path, path, headered=False))

    def test_rejects_wrong_header_exactly(self, tmp_path: Path) -> None:
        path = tmp_path / "positions.csv"
        path.write_text("barcode,in_tissue,x\na,1,2\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="exact columns"):
            read_positions(_layout(tmp_path, path, headered=True))


class TestPrepareTable:
    """Align count order while preserving CSR-array sparsity and extra position rows."""

    def test_aligns_positions_and_populates_spatial_metadata(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "positions.csv"
        path.write_text(_position_rows(), encoding="utf-8")
        layout = _layout(tmp_path, path, headered=False)
        positions = read_positions(layout)
        adata = AnnData(
            X=sparse.csr_array(np.asarray([[1, 0], [0, 2]], dtype=np.float32)),
            obs=pd.DataFrame(index=pd.Index(["a", "b"], dtype="string")),
            var=pd.DataFrame(index=pd.Index(["gene-a", "gene-b"], dtype="string")),
        )
        mocker.patch("spatialdata_io.readers.visium._tables._read_counts", return_value=adata)

        result = prepare_table(layout, positions, options=_options())

        assert sparse.issparse(result.adata.X)
        assert isinstance(result.adata.X, sparse.sparray)
        assert not isinstance(result.adata.X, sparse.spmatrix)
        assert result.positions.index.tolist() == ["a", "b"]
        assert result.adata.obs["spot_id"].tolist() == ["a", "b"]
        assert result.adata.obs["region"].tolist() == ["sample", "sample"]
        np.testing.assert_array_equal(result.adata.obsm["spatial"], np.asarray([[20.0, 10.0], [40.5, 30.5]]))
        assert result.adata.uns["spatial"] == {
            "sample": {
                "metadata": {
                    "chemistry_description": "chemistry",
                    "software_version": "spaceranger-2.1.0",
                },
                "scalefactors": {"spot_diameter_fullres": 6.0},
            }
        }

    def test_rejects_count_barcode_missing_from_positions(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "positions.csv"
        path.write_text("a,1,0,0,1,2\n", encoding="utf-8")
        layout = _layout(tmp_path, path, headered=False)
        adata = AnnData(
            X=sparse.csr_array((2, 1), dtype=np.float32),
            obs=pd.DataFrame(index=["a", "missing"]),
        )
        mocker.patch("spatialdata_io.readers.visium._tables._read_counts", return_value=adata)

        with pytest.raises(ReaderFormatError, match="absent from tissue positions"):
            prepare_table(layout, read_positions(layout), options=_options())


class TestLinkTable:
    """Parse linkage once against an independent materialized shape index."""

    def test_links_exact_barcode_membership(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "positions.csv"
        path.write_text("a,1,0,0,1,2\n", encoding="utf-8")
        layout = _layout(tmp_path, path, headered=False)
        adata = AnnData(
            X=sparse.csr_array([[1]], dtype=np.float32),
            obs=pd.DataFrame(index=["a"]),
            var=pd.DataFrame(index=["gene"]),
        )
        mocker.patch("spatialdata_io.readers.visium._tables._read_counts", return_value=adata)
        prepared = prepare_table(layout, read_positions(layout), options=_options())

        result = link_table(prepared, dataset_id="sample", shape_index=pd.Index(["a"], name="spot_id"))

        assert result is adata
        assert get_table_keys(result) == ("sample", "region", "spot_id")

    def test_rejects_mismatched_shape_membership(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "positions.csv"
        path.write_text("a,1,0,0,1,2\n", encoding="utf-8")
        layout = _layout(tmp_path, path, headered=False)
        adata = AnnData(X=sparse.csr_array([[1]]), obs=pd.DataFrame(index=["a"]), var=pd.DataFrame(index=["g"]))
        mocker.patch("spatialdata_io.readers.visium._tables._read_counts", return_value=adata)
        prepared = prepare_table(layout, read_positions(layout), options=_options())

        with pytest.raises(ReaderFormatError, match="target-instance membership"):
            link_table(prepared, dataset_id="sample", shape_index=pd.Index(["other"]))
