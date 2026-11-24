"""Discovery and region identity for PhenoCycler layouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.phenocycler._layout import discover_layout, normalize_regions

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class TestDiscoverLayout:
    def test_retains_scan_suffix_when_needed_and_normalizes_codex_region(self, tmp_path: Path) -> None:
        names = ("dataset_REG0007.qptiff", "A_Scan1.qptiff", "A_scan2.qptiff")
        for name in names:
            (tmp_path / name).touch()
        layout = discover_layout(tmp_path, regions=None, read_all_bit_depths=False, path_policy="strict")
        assert [(entry.scan.name, entry.scan.region) for entry in layout.regions] == [
            ("A_Scan1.qptiff", "A_Scan1"),
            ("A_scan2.qptiff", "A_scan2"),
            ("dataset_REG0007.qptiff", "reg007"),
        ]

    @pytest.mark.parametrize(
        ("paths", "match"),
        [
            (("nothing.txt",), "no processed QPTIFF"),
            (("a_reg1.QPTIFF", "b_reg001.qptiff"), "multiple scans mapping to region names"),
            (("A.qptiff", "a.qptiff"), "same bit-depth rank"),
            (("A.qptiff", "A.er.qptiff", "A_8bit.qptiff"), "colliding with region names"),
            (("A.qptiff", "A.xpd", "A.qptiff.xpd"), "multiple sidecars"),
        ],
    )
    def test_ambiguous_candidates_fail(self, tmp_path: Path, paths: tuple[str, ...], match: str) -> None:
        for name in paths:
            (tmp_path / name).touch()
        with pytest.raises(ReaderFormatError, match=match):
            discover_layout(tmp_path, regions=None, read_all_bit_depths=True, path_policy="strict")

    def test_reports_missing_file_wrong_suffix_directory_named_qpi(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"absent.qptiff"):
            discover_layout(tmp_path / "absent.qptiff", regions=None, read_all_bit_depths=False, path_policy="strict")
        unsupported = tmp_path / "unsupported.png"
        unsupported.touch()
        with pytest.raises(ReaderFormatError, match=r"QPI \.tif"):
            discover_layout(unsupported, regions=None, read_all_bit_depths=False, path_policy="strict")
        (tmp_path / "image.qptiff").mkdir()
        with pytest.raises(IsADirectoryError, match=r"image.qptiff"):
            discover_layout(tmp_path, regions=None, read_all_bit_depths=False, path_policy="strict")

    def test_selection_error_has_available_names(self, tmp_path: Path) -> None:
        (tmp_path / "name.qptiff").touch()
        with pytest.raises(ValueError, match=r"Unknown regions.*Available regions.*name"):
            discover_layout(tmp_path, regions=("missing",), read_all_bit_depths=False, path_policy="strict")

    def test_path_policy_and_symlink_cycle(self, tmp_path: Path) -> None:
        external = tmp_path / "data"
        external.mkdir()
        (external / "sample.qptiff").touch()
        root = tmp_path / "dataset"
        root.mkdir()
        (root / "linked").symlink_to(external, target_is_directory=True)
        with pytest.raises(ReaderFormatError, match="outside"):
            discover_layout(root, regions=None, read_all_bit_depths=False, path_policy="strict")
        result = discover_layout(root, regions=None, read_all_bit_depths=False, path_policy="relaxed")
        assert result.regions[0].scan.region == "sample"

        (root / "recursion").symlink_to(root, target_is_directory=True)
        with pytest.raises(ReaderFormatError, match="recursive directory symlink"):
            discover_layout(root, regions=None, read_all_bit_depths=False, path_policy="relaxed")

    def test_use_symlink_name_to_find_sidecar(self, tmp_path: Path) -> None:
        target = tmp_path / "payload"
        target.touch()
        (tmp_path / "sample.qptiff").symlink_to(target)
        sidecar = tmp_path / "sample.xpd"
        sidecar.touch()
        result = discover_layout(tmp_path, regions=None, read_all_bit_depths=False, path_policy="strict")
        assert result.regions[0].xpd == sidecar
        assert result.regions[0].scan.name == "sample.qptiff"

    def test_documented_hubmap_sidecar_needs_one_corresponding_scan(self, tmp_path: Path) -> None:
        images = tmp_path / "raw" / "images"
        images.mkdir(parents=True)
        (images / "sample.qptiff").touch()
        (images / "sample.er.qptiff").touch()
        processed = tmp_path / "lab_processed" / "images"
        processed.mkdir(parents=True)
        generic = processed / "qptiff.channels.csv"
        generic.touch()
        result = discover_layout(tmp_path, regions=None, read_all_bit_depths=False, path_policy="strict")
        assert result.regions[0].channels_csv == generic

        (images / "another.qptiff").touch()
        with pytest.raises(ReaderFormatError, match="1 generic channels CSV files for 2 scans"):
            discover_layout(tmp_path, regions=("sample",), read_all_bit_depths=False, path_policy="strict")

        specific = processed / "sample.qptiff.channels.csv"
        specific.touch()
        scoped = discover_layout(tmp_path, regions=("sample",), read_all_bit_depths=False, path_policy="strict")
        assert scoped.regions[0].channels_csv == specific


class TestNormalizeRegions:
    @pytest.mark.parametrize(
        ("value", "error", "message"),
        [
            ("region", TypeError, "sequence"),
            ((1,), TypeError, "contain strings"),
            ((), ValueError, "at least one"),
            ((" ",), ValueError, "nonempty"),
            (("A", "a"), ValueError, "Repeated region"),
        ],
    )
    def test_invalid_options(self, value: object, error: type[Exception], message: str) -> None:
        with pytest.raises(error, match=message):
            normalize_regions(cast("Sequence[str]", value))

    def test_sanitizes_names_and_accepts_codex_tokens(self) -> None:
        with pytest.warns(UserWarning, match=r"My slide.*My_slide"):
            assert normalize_regions(("reg1", "My slide")) == ("reg001", "My_slide")
