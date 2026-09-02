"""Tests for deterministic Visium layout discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.visium._layout import H5Counts, MtxCounts, VisiumLayout, discover_layout

if TYPE_CHECKING:
    from pathlib import Path


def _write_required_spatial(root: Path, *, current: bool = True) -> None:
    spatial = root / "spatial"
    spatial.mkdir()
    name = "tissue_positions.csv" if current else "tissue_positions_list.csv"
    header = "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n" if current else ""
    (spatial / name).write_text(f"{header}a,1,0,0,1,2\n", encoding="utf-8")
    (spatial / "scalefactors_json.json").write_text("{}", encoding="utf-8")


def _discover(root: Path, **overrides: str | Path | None) -> VisiumLayout:
    return discover_layout(
        root,
        counts_source=overrides.get("counts_source"),
        fullres_image_file=overrides.get("fullres_image_file"),
        positions_file=overrides.get("positions_file"),
        scale_factors_file=overrides.get("scale_factors_file"),
    )


def _write_mex(path: Path, *, prefix: str = "", compressed: bool = True) -> None:
    path.mkdir()
    gzip_suffix = ".gz" if compressed else ""
    for name in ("matrix.mtx", "barcodes.tsv", "features.tsv"):
        (path / f"{prefix}{name}{gzip_suffix}").touch()


class TestDiscoverLayout:
    """Exercise count, position, image, and override discovery as one transaction."""

    def test_standard_h5_precedes_standard_mex_and_finds_optional_images(self, tmp_path: Path) -> None:
        _write_required_spatial(tmp_path)
        h5 = tmp_path / "filtered_feature_bc_matrix.h5"
        h5.touch()
        _write_mex(tmp_path / "filtered_feature_bc_matrix")
        for name in ("tissue_hires_image.png", "tissue_lowres_image.png"):
            (tmp_path / "spatial" / name).touch()

        layout = _discover(tmp_path)

        assert isinstance(layout.counts, H5Counts)
        assert layout.counts.path == h5
        assert layout.counts.filename_prefix is None
        assert layout.position_format == "headered"
        assert layout.hires_image == tmp_path / "spatial" / "tissue_hires_image.png"
        assert layout.lowres_image == tmp_path / "spatial" / "tissue_lowres_image.png"

    @pytest.mark.parametrize("compressed", [False, True])
    def test_standard_mex_records_storage_variant(self, tmp_path: Path, *, compressed: bool) -> None:
        _write_required_spatial(tmp_path)
        _write_mex(tmp_path / "filtered_feature_bc_matrix", prefix="sample_", compressed=compressed)

        counts = _discover(tmp_path).counts

        assert isinstance(counts, MtxCounts)
        assert counts.prefix == "sample_"
        assert counts.compressed is compressed
        assert counts.filename_prefix is None

    @pytest.mark.parametrize("kind", ["h5", "mex"])
    def test_legacy_prefixed_count_source_infers_dataset_prefix(self, tmp_path: Path, kind: str) -> None:
        _write_required_spatial(tmp_path, current=False)
        if kind == "h5":
            (tmp_path / "library_filtered_feature_bc_matrix.h5").touch()
        else:
            _write_mex(tmp_path / "library_filtered_feature_bc_matrix")

        counts = _discover(tmp_path).counts

        assert counts.filename_prefix == "library"
        assert _discover(tmp_path).position_format == "headerless"

    def test_relative_overrides_resolve_under_root(self, tmp_path: Path) -> None:
        _write_required_spatial(tmp_path)
        (tmp_path / "raw_feature_bc_matrix.h5").touch()
        (tmp_path / "positions.csv").write_text(
            "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\na,1,0,0,1,2\n",
            encoding="utf-8",
        )
        (tmp_path / "scales.json").touch()
        (tmp_path / "full.tif").touch()

        layout = _discover(
            tmp_path,
            counts_source="raw_feature_bc_matrix.h5",
            positions_file="positions.csv",
            scale_factors_file="scales.json",
            fullres_image_file="full.tif",
        )

        assert layout.counts.path == tmp_path / "raw_feature_bc_matrix.h5"
        assert layout.positions == tmp_path / "positions.csv"
        assert layout.scale_factors == tmp_path / "scales.json"
        assert layout.fullres_image == tmp_path / "full.tif"

    def test_current_positions_take_precedence_over_legacy_file(self, tmp_path: Path) -> None:
        _write_required_spatial(tmp_path)
        (tmp_path / "filtered_feature_bc_matrix.h5").touch()
        (tmp_path / "spatial" / "tissue_positions_list.csv").write_text("a,1,0,0,1,2\n", encoding="utf-8")

        layout = _discover(tmp_path)

        assert layout.positions.name == "tissue_positions.csv"
        assert layout.position_format == "headered"

    def test_detects_header_in_legacy_position_filename(self, tmp_path: Path) -> None:
        _write_required_spatial(tmp_path, current=False)
        (tmp_path / "filtered_feature_bc_matrix.h5").touch()
        path = tmp_path / "spatial" / "tissue_positions_list.csv"
        path.write_text(
            "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\na,1,0,0,1,2\n",
            encoding="utf-8",
        )

        layout = _discover(tmp_path)

        assert layout.positions == path
        assert layout.position_format == "headered"

    def test_rejects_ambiguous_legacy_count_sources(self, tmp_path: Path) -> None:
        _write_required_spatial(tmp_path)
        (tmp_path / "sample_filtered_feature_bc_matrix.h5").touch()
        _write_mex(tmp_path / "sample_filtered_feature_bc_matrix")

        with pytest.raises(ReaderFormatError, match=r"both .* and"):
            _discover(tmp_path)

    def test_rejects_multiple_complete_mex_triplets(self, tmp_path: Path) -> None:
        _write_required_spatial(tmp_path)
        mex = tmp_path / "filtered_feature_bc_matrix"
        _write_mex(mex, prefix="a_")
        for name in ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"):
            (mex / f"b_{name}").touch()

        with pytest.raises(ReaderFormatError, match="multiple MEX triplets"):
            _discover(tmp_path)

    def test_rejects_incomplete_mex_triplet(self, tmp_path: Path) -> None:
        _write_required_spatial(tmp_path)
        mex = tmp_path / "filtered_feature_bc_matrix"
        mex.mkdir()
        (mex / "matrix.mtx.gz").touch()

        with pytest.raises(ReaderFormatError, match="no complete MEX triplet"):
            _discover(tmp_path)

    def test_rejects_mixed_compression_mex_triplet(self, tmp_path: Path) -> None:
        _write_required_spatial(tmp_path)
        mex = tmp_path / "filtered_feature_bc_matrix"
        mex.mkdir()
        (mex / "matrix.mtx").touch()
        (mex / "barcodes.tsv.gz").touch()
        (mex / "features.tsv.gz").touch()

        with pytest.raises(ReaderFormatError, match="consistently compressed or uncompressed"):
            _discover(tmp_path)

    def test_missing_standard_counts_preserves_precise_path_error(self, tmp_path: Path) -> None:
        _write_required_spatial(tmp_path)

        with pytest.raises(FileNotFoundError) as error:
            _discover(tmp_path)

        assert error.value.filename == str(tmp_path / "filtered_feature_bc_matrix.h5")

    def test_bounds_position_header_inspection(self, tmp_path: Path) -> None:
        _write_required_spatial(tmp_path)
        (tmp_path / "filtered_feature_bc_matrix.h5").touch()
        (tmp_path / "spatial" / "tissue_positions.csv").write_bytes(b"x" * 1_025 + b"\n")

        with pytest.raises(ReaderFormatError, match="first row longer than 1024 bytes"):
            _discover(tmp_path)
