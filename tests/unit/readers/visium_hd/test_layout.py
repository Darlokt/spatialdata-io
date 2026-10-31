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

    def test_root_mappings(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        nested = tmp_path / "segmented_outputs/barcode_mappings.parquet"
        nested.rename(tmp_path / "abc_barcode_mappings.parquet")
        layout = discover_layout(tmp_path, bin_sizes=[], include_cell_boundaries=False, include_nucleus_table=True)
        assert layout.nucleus_counts is not None
        assert layout.nucleus_counts.barcode_mappings.name == "abc_barcode_mappings.parquet"

    @pytest.mark.parametrize(
        "root_images", [None, (), ("hires",), ("lowres", "cytassist"), ("hires", "lowres", "cytassist")]
    )
    def test_per_artifact_image_precedence(self, tmp_path: Path, root_images: tuple[str, ...] | None) -> None:
        write_bundle(tmp_path, images=True)
        names = {
            "hires": "tissue_hires_image.png",
            "lowres": "tissue_lowres_image.png",
            "cytassist": "cytassist_image.tiff",
        }
        spatial = tmp_path / "spatial"
        if root_images is not None:
            spatial.mkdir()
            for key in root_images:
                (spatial / names[key]).touch()
        layout = discover_layout(tmp_path, include_cytassist_image=True)
        assert layout.images is not None
        for key, name in names.items():
            directory = (
                spatial
                if root_images is not None and key in root_images
                else tmp_path / "binned_outputs/square_002um/spatial"
            )
            assert getattr(layout.images, key) == directory / name

    @pytest.mark.parametrize("name", ["tissue_hires_image.png", "tissue_lowres_image.png", "cytassist_image.tiff"])
    @pytest.mark.parametrize("kind", ["directory", "dangling"])
    def test_invalid_root_image_prevents_fallback(self, tmp_path: Path, name: str, kind: str) -> None:
        write_bundle(tmp_path, images=True)
        spatial = tmp_path / "spatial"
        spatial.mkdir()
        invalid = spatial / name
        if kind == "directory":
            invalid.mkdir()
        else:
            invalid.symlink_to(spatial / "missing")
        expected = IsADirectoryError if kind == "directory" else FileNotFoundError
        with pytest.raises(expected) as error:
            discover_layout(tmp_path, include_cytassist_image=True)
        assert error.value.filename == str(invalid)

    def test_cytassist_only_skips_unrequested_images(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, images=True)
        spatial = tmp_path / "spatial"
        spatial.mkdir()
        (spatial / "tissue_hires_image.png").mkdir()
        layout = discover_layout(tmp_path, include_images=False, include_cytassist_image=True)
        assert layout.images is not None
        assert layout.images.hires is None
        assert layout.images.lowres is None
        assert layout.images.fullres is None
        assert layout.images.cytassist == tmp_path / "binned_outputs/square_002um/spatial/cytassist_image.tiff"

    @pytest.mark.parametrize("kind", ["file", "dangling"])
    def test_invalid_root_image_directory(self, tmp_path: Path, kind: str) -> None:
        write_bundle(tmp_path, images=True)
        spatial = tmp_path / "spatial"
        if kind == "file":
            spatial.touch()
        else:
            spatial.symlink_to(tmp_path / "missing")
        expected = NotADirectoryError if kind == "file" else FileNotFoundError
        with pytest.raises(expected) as error:
            discover_layout(tmp_path)
        assert error.value.filename == str(spatial)

    def test_missing_requested_cytassist(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        (tmp_path / "spatial").mkdir()
        with pytest.raises(FileNotFoundError, match="cytassist_image"):
            discover_layout(tmp_path, include_cytassist_image=True)

    def test_no_selected_counts_uses_feature_slice(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, images=True)
        for matrix in tmp_path.glob("binned_outputs/*/*feature_bc_matrix.h5"):
            matrix.unlink()
        layout = discover_layout(
            tmp_path, bin_sizes=(), include_cell_boundaries=False, include_nucleus_boundaries=False
        )
        assert layout.bins == ()
        assert layout.segmentation is None
        assert layout.feature_slice == tmp_path / "feature_slice.h5"

    def test_no_counts_requires_feature_even_with_unselected_matrix(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        (tmp_path / "feature_slice.h5").unlink()
        with pytest.raises(FileNotFoundError, match="feature_slice"):
            discover_layout(tmp_path, bin_sizes=(), include_cell_boundaries=False, include_nucleus_boundaries=False)

    def test_selected_cell_counts_need_no_feature_or_bin_counts(self, tmp_path: Path) -> None:
        write_bundle(tmp_path)
        (tmp_path / "feature_slice.h5").unlink()
        for matrix in tmp_path.glob("binned_outputs/*/*feature_bc_matrix.h5"):
            matrix.unlink()
        layout = discover_layout(tmp_path, bin_sizes=(), include_images=False)
        assert layout.feature_slice is None
        assert layout.segmentation is not None
        assert layout.segmentation.counts is not None

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
