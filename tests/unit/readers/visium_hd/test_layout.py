"""Requested-product discovery and artifact ambiguity."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.visium_hd._layout import discover_layout
from tests.integration.readers.visium_hd._fixtures import write_bundle


class TestDiscoverLayout:
    """Verify discover layout contracts."""

    @pytest.mark.parametrize(
        ("version", "prefix", "name"), [("3.0.0", "", "hd_feature_slice.h5"), ("4.1.0", "dataset_", "feature_slice.h5")]
    )
    def test_layouts(
        self, tmp_path: Path, version: str, prefix: str, name: Literal["feature_slice.h5", "hd_feature_slice.h5"]
    ) -> None:
        write_bundle(
            tmp_path, version=version, prefix=prefix, feature_slice_name=name, segmented=version.startswith("4")
        )
        layout = discover_layout(tmp_path, include_images=False)
        assert [item.size_um for item in layout.bins] == [2, 8, 16]
        assert layout.feature_slice is not None
        assert layout.feature_slice.name == prefix + name
        assert (layout.segmentation is not None) == version.startswith("4")
        assert layout.images is None
        assert layout.nucleus_counts is None

    def test_selected_raw_and_nested_mappings(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, sizes=(2, 6), prefix="abc_")
        layout = discover_layout(
            tmp_path, bin_sizes=[6], bin_counts_kind="raw", cell_counts_kind="raw", include_nucleus_table=True
        )
        assert layout.bins[0].counts.name == "raw_feature_bc_matrix.h5"
        assert layout.segmentation is not None
        assert layout.segmentation.counts is not None
        assert layout.segmentation.counts.name == "raw_feature_cell_matrix.h5"
        assert layout.nucleus_counts is not None
        assert layout.nucleus_counts.barcode_mappings.parent.name == "segmented_outputs"
        assert layout.nucleus_counts.raw_two_micron_counts.name == "raw_feature_bc_matrix.h5"

    def test_root_mappings_and_identity_fallback(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        nested = tmp_path / "segmented_outputs/barcode_mappings.parquet"
        nested.rename(tmp_path / "abc_barcode_mappings.parquet")
        layout = discover_layout(tmp_path, bin_sizes=[], include_cell_boundaries=False, include_nucleus_table=True)
        assert layout.nucleus_counts is not None
        assert layout.nucleus_counts.barcode_mappings.name == "abc_barcode_mappings.parquet"
        images = discover_layout(
            tmp_path, bin_sizes=[], include_cell_boundaries=False, include_nucleus_boundaries=False
        )
        assert images.identity_file.name == "filtered_feature_bc_matrix.h5"

    @pytest.mark.parametrize(
        "mode",
        [
            "sizes",
            "feature",
            "duplicate-bin",
            "odd-bin",
            "mapping",
            "missing-mapping",
            "missing-base",
            "missing-segment",
            "missing-binned",
        ],
    )
    def test_errors(self, tmp_path: Path, mode: str) -> None:
        write_bundle(tmp_path)
        if mode == "sizes":
            pass
        elif mode == "feature":
            (tmp_path / "other_feature_slice.h5").touch()
        elif mode == "duplicate-bin":
            (tmp_path / "binned_outputs/square_0002um").mkdir()
        elif mode == "odd-bin":
            (tmp_path / "binned_outputs/square_003um").mkdir()
        elif mode == "mapping":
            (tmp_path / "barcode_mappings.parquet").touch()
        elif mode == "missing-mapping":
            (tmp_path / "segmented_outputs/barcode_mappings.parquet").unlink()
        elif mode == "missing-base":
            (tmp_path / "binned_outputs/square_002um").rename(tmp_path / "base")
        elif mode == "missing-binned":
            (tmp_path / "binned_outputs").rename(tmp_path / "unused")
        elif mode == "missing-segment":
            (tmp_path / "segmented_outputs").rename(tmp_path / "unused")
        with pytest.raises((ReaderFormatError, FileNotFoundError)):
            discover_layout(
                tmp_path,
                bin_sizes=[6] if mode == "sizes" else "all",
                include_nucleus_table=mode in {"mapping", "missing-mapping"},
                include_cell_boundaries=True if mode == "missing-segment" else None,
            )

    def test_skips_unrequested_and_decoys(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        (tmp_path / "segmented_outputs/cell_segmentations.geojson").unlink()
        (tmp_path / "microscope_image").write_text("invalid but unrequested")
        (tmp_path / "binned_outputs/decoy").mkdir()
        layout = discover_layout(
            tmp_path,
            bin_sizes=[],
            include_cell_boundaries=False,
            include_nucleus_boundaries=False,
            include_images=False,
        )
        assert layout.segmentation is None
        assert layout.images is None

    @pytest.mark.parametrize("fullres", ["root", "folder"])
    def test_fullres_auto(self, tmp_path: Path, fullres: Literal["root", "folder"]) -> None:
        write_bundle(tmp_path, fullres=fullres)
        layout = discover_layout(tmp_path)
        assert layout.images is not None
        assert layout.images.fullres is not None
        assert layout.images.fullres.suffix == ".tif"

    def test_fullres_ambiguity_override_and_root_images(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, fullres="root")
        (tmp_path / "another_tissue_image.tiff").touch()
        with pytest.raises(ReaderFormatError, match="full-resolution"):
            discover_layout(tmp_path)
        spatial = tmp_path / "spatial"
        spatial.mkdir()
        (spatial / "cytassist_image.tiff").touch()
        for source in (Path("tissue_image.tif"), tmp_path / "tissue_image.tif"):
            layout = discover_layout(tmp_path, fullres_image_file=source, include_cytassist_image=True)
            assert layout.images is not None
            assert layout.images.fullres is not None
            assert layout.images.cytassist == spatial / "cytassist_image.tiff"

    def test_bins_without_feature_slice(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        (tmp_path / "feature_slice.h5").unlink()
        assert discover_layout(tmp_path, include_images=False).feature_slice is None
        with pytest.raises(FileNotFoundError):
            discover_layout(tmp_path, include_cytassist_image=True)
