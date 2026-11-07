"""Tests for Visium position parsing, sparse alignment, and linkage."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from spatialdata.models import get_table_keys

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium._layout import H5Counts, VisiumLayout
from spatialdata_io.readers.visium._metadata import CountMetadata, ScaleFactors
from spatialdata_io.readers.visium._schemas import POSITION_COLUMNS
from spatialdata_io.readers.visium._tables import link_table, prepare_table, read_positions

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_NO_METADATA = CountMetadata(None, None, None)


def _layout(root: Path, positions: Path, *, headered: bool, metadata: CountMetadata = _NO_METADATA) -> VisiumLayout:
    counts = root / "counts.h5"
    counts.touch(exist_ok=True)
    scales = root / "scales.json"
    scales.touch(exist_ok=True)
    return VisiumLayout(
        counts=H5Counts(ArtifactFile(counts), None, metadata),
        positions=ArtifactFile(positions),
        position_format="headered" if headered else "headerless",
        scale_factors=ArtifactFile(scales),
        hires_image=None,
        lowres_image=None,
        fullres_image=None,
    )


def _position_rows() -> str:
    return "b,0,1,2,30.5,40.5\na,1,0,3,10,20\nextra,1,2,1,50,60\n"


class TestReadPositions:
    """Validate both vendor CSV schemas and all coordinate columns."""

    @pytest.mark.parametrize("headered", [False, True])
    @pytest.mark.parametrize("barcode", ["a,b", 'a"b', "a\nb", "NA", "NULL"])
    def test_preserves_quoted_and_literal_barcodes(self, tmp_path: Path, *, headered: bool, barcode: str) -> None:
        """Keep source barcode text, escaped quotes, embedded newlines, and literal null-like words."""
        path = tmp_path / "positions.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            if headered:
                writer.writerow(POSITION_COLUMNS)
            writer.writerow([barcode, "1", "9007199254740993", "2", "0.10000000000000002", "20"])

        result = read_positions(_layout(tmp_path, path, headered=headered))

        assert result.index.tolist() == [barcode]
        assert result["array_row"].tolist() == [2**53 + 1]
        assert result["array_col"].tolist() == [2]
        assert result["pxl_row_in_fullres"].tolist() == [float("0.10000000000000002")]

    @pytest.mark.parametrize("newline_offset", [(1 << 20) - 1, 1 << 20], ids=["before-block", "at-block"])
    def test_preserves_multiline_barcode_across_default_arrow_block(self, tmp_path: Path, newline_offset: int) -> None:
        """Preserve identities at boundaries that can otherwise truncate or split quoted records."""
        path = tmp_path / "positions.csv"
        header = (",".join((*POSITION_COLUMNS, "extension")) + "\n").encode()
        record = b"spot-00000000,1,2,3,4.25,5.125," + b"x" * 96 + b"\n"
        barcode = "m" * 40 + "\n" + "l" * 80
        quote_offset = newline_offset - 41
        count = (quote_offset - len(header)) // len(record) - 1
        barcodes = [f"spot-{index:08d}" for index in range(count)]
        prefix = header + b"".join(name.encode() + b",1,2,3,4.25,5.125," + b"x" * 96 + b"\n" for name in barcodes)
        padding = quote_offset - len(prefix) - len(b"pad,1,2,3,4,5,\n")
        prefix += b"pad,1,2,3,4,5," + b"x" * padding + b"\n"
        assert len(prefix) == quote_offset
        source = prefix + b'"' + barcode.encode() + b'",1,2,3,4.25,5.125,tail\n'
        assert source.find(b"\n", quote_offset) == newline_offset
        path.write_bytes(source)

        result = read_positions(_layout(tmp_path, path, headered=True))

        assert result.index.tolist() == [*barcodes, "pad", barcode]
        assert "extension" not in result.columns
        np.testing.assert_array_equal(result.loc[barcode, ["array_row", "array_col"]], [2, 3])

    @pytest.mark.parametrize("row", ["a,1,0,0,10,20", "a,1,0,0,10,20,extra,surplus"])
    def test_rejects_inconsistent_record_widths(self, tmp_path: Path, row: str) -> None:
        """Require complete source records before projecting supported position fields."""
        path = tmp_path / "positions.csv"
        path.write_text(",".join((*POSITION_COLUMNS, "quality")) + "\n" + row + "\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="record widths"):
            read_positions(_layout(tmp_path, path, headered=True))

    @pytest.mark.parametrize("headered", [False, True])
    def test_reads_schema_without_reordering_rows(self, tmp_path: Path, *, headered: bool) -> None:
        """Reads schema without reordering rows."""
        path = tmp_path / "positions.csv"
        header = "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n" if headered else ""
        path.write_text(header + _position_rows(), encoding="utf-8")

        result = read_positions(_layout(tmp_path, path, headered=headered))

        assert result.index.tolist() == ["b", "a", "extra"]
        assert str(result["in_tissue"].dtype) == "int64[pyarrow]"
        assert result["array_row"].tolist() == [1, 0, 2]
        np.testing.assert_array_equal(result["pxl_col_in_fullres"], np.asarray([40.5, 20.0, 60.0]))

    def test_rejects_missing_required_header_fields(self, tmp_path: Path) -> None:
        """Rejects missing required header fields."""
        path = tmp_path / "positions.csv"
        path.write_text("barcode,in_tissue,x\na,1,2\n", encoding="utf-8")

        with pytest.raises(ReaderFormatError, match="required columns"):
            read_positions(_layout(tmp_path, path, headered=True))

    def test_source_failure_keeps_its_error_with_context(self, tmp_path: Path) -> None:
        path = tmp_path / "positions.csv"
        path.mkdir()

        with pytest.raises(IsADirectoryError) as caught:
            read_positions(_layout(tmp_path, path, headered=True))

        assert any("tissue positions" in note for note in caught.value.__notes__)


class TestPrepareTable:
    """Align count order while preserving CSR-array sparsity and extra position rows."""

    def test_aligns_positions_and_populates_spatial_metadata(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "positions.csv"
        path.write_text(_position_rows(), encoding="utf-8")
        layout = _layout(
            tmp_path, path, headered=False, metadata=CountMetadata("sample", "chemistry", "spaceranger-2.1.0")
        )
        positions = read_positions(layout)
        adata = AnnData(
            X=sparse.csr_array(np.asarray([[1, 0], [0, 2]], dtype=np.float32)),
            obs=pd.DataFrame(index=pd.Index(["a", "b"], dtype="string")),
            var=pd.DataFrame(index=pd.Index(["gene-a", "gene-b"], dtype="string")),
        )
        mocker.patch("spatialdata_io.readers.visium._tables._read_counts", return_value=adata)

        result = prepare_table(
            layout,
            positions,
            dataset_id="sample",
            pixel_frame="sample_pixels",
            scale_factors=ScaleFactors(0.5, 0.25, 6.0, {"spot_diameter_fullres": 6.0}),
            make_var_names_unique=True,
        )

        assert sparse.issparse(result.adata.X)
        assert isinstance(result.adata.X, sparse.sparray)
        assert not isinstance(result.adata.X, sparse.spmatrix)

        assert result.positions.index.tolist() == ["a", "b"]
        assert result.adata.obs["spot_id"].tolist() == ["a", "b"]
        assert result.adata.obs["region"].tolist() == ["sample", "sample"]
        np.testing.assert_array_equal(result.adata.obsm["spatial_pixels"], np.asarray([[20.0, 10.0], [40.5, 30.5]]))

        assert result.adata.uns["spatial"] == {
            "sample": {
                "metadata": {
                    "chemistry_description": "chemistry",
                    "software_version": "spaceranger-2.1.0",
                },
                "scalefactors": {"spot_diameter_fullres": 6.0},
            }
        }
        assert result.adata.uns["spatialdata_io_positions"]["coordinates"]["spatial_pixels"] == {
            "axes": ["x", "y"],
            "units": "pixel",
            "frame": "sample_pixels",
            "position": "full-resolution spot centre",
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
            prepare_table(
                layout,
                read_positions(layout),
                dataset_id="sample",
                pixel_frame="sample_pixels",
                scale_factors=ScaleFactors(0.5, 0.25, 6.0, {}),
                make_var_names_unique=True,
            )

    def test_rejects_duplicate_count_barcodes(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "positions.csv"
        path.write_text("a,1,0,0,1,2\n", encoding="utf-8")
        layout = _layout(tmp_path, path, headered=False)
        with pytest.warns(UserWarning, match="Observation names are not unique"):
            adata = AnnData(X=sparse.csr_array((2, 1), dtype=np.float32), obs=pd.DataFrame(index=["a", "a"]))
        mocker.patch("spatialdata_io.readers.visium._tables._read_counts", return_value=adata)

        with pytest.raises(ReaderFormatError, match=r"duplicate observation barcodes \('a',\)") as caught:
            prepare_table(
                layout,
                read_positions(layout),
                dataset_id="sample",
                pixel_frame="sample_pixels",
                scale_factors=ScaleFactors(0.5, 0.25, 6.0, {}),
                make_var_names_unique=True,
            )

        assert caught.value.path == layout.counts.path
        assert caught.value.context.artifact == "count matrix"


class TestLinkTable:
    """Parse linkage once against an independent materialized shape index."""

    def test_count_precision_error_identifies_source_and_version(self, tmp_path: Path) -> None:
        layout = _layout(
            tmp_path,
            tmp_path / "positions.csv",
            headered=False,
            metadata=CountMetadata(None, None, "spaceranger-2.1.0"),
        )
        table = AnnData(sparse.csr_array([[2**53 + 1]], dtype=np.int64))

        with pytest.raises(ReaderFormatError, match="not exactly representable") as caught:
            link_table(table, dataset_id="sample", shape_index=pd.Index(["0"]), layout=layout)

        assert caught.value.path == layout.counts.path
        assert caught.value.context == ReaderErrorContext("visium", "count matrix", "2.1.0")

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
        prepared = prepare_table(
            layout,
            read_positions(layout),
            dataset_id="sample",
            pixel_frame="sample_pixels",
            scale_factors=ScaleFactors(0.5, 0.25, 6.0, {}),
            make_var_names_unique=True,
        )

        result = link_table(
            prepared.adata,
            dataset_id="sample",
            shape_index=pd.Index(["a"], name="spot_id"),
            layout=layout,
        )

        assert result is adata
        assert get_table_keys(result) == ("sample", "region", "spot_id")

    def test_rejects_mismatched_shape_membership(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "positions.csv"
        path.write_text("a,1,0,0,1,2\n", encoding="utf-8")
        layout = _layout(tmp_path, path, headered=False)
        adata = AnnData(X=sparse.csr_array([[1]]), obs=pd.DataFrame(index=["a"]), var=pd.DataFrame(index=["g"]))
        mocker.patch("spatialdata_io.readers.visium._tables._read_counts", return_value=adata)
        prepared = prepare_table(
            layout,
            read_positions(layout),
            dataset_id="sample",
            pixel_frame="sample_pixels",
            scale_factors=ScaleFactors(0.5, 0.25, 6.0, {}),
            make_var_names_unique=True,
        )

        with pytest.raises(ReaderFormatError, match="target-instance membership"):
            link_table(prepared.adata, dataset_id="sample", shape_index=pd.Index(["other"]), layout=layout)
