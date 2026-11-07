"""Steinbock deterministic pairing, artifact validation, and containment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.steinbock._layout import discover_layout, validate_masks_dir, validate_table_file

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import PathPolicy


def _write_paths(root: Path) -> None:
    """Create artifact paths without involving raster or table parsers."""
    (root / "ome").mkdir()
    (root / "cells.h5ad").touch()
    for directory in ("masks", "masks_ilastik"):
        (root / directory).mkdir()
        for name in ("sample-b", "sample-a"):
            (root / directory / f"{name}.tiff").touch()
    for name in ("sample-b", "sample-a"):
        (root / "ome" / f"{name}.ome.tiff").touch()


class TestDiscoverLayout:
    """Require a complete paired layout with contained validated artifacts."""

    @pytest.mark.parametrize("masks_dir", ["masks", "masks_ilastik"])
    def test_exact_pairing_and_deterministic_order(self, tmp_path: Path, masks_dir: str) -> None:
        _write_paths(tmp_path)
        (tmp_path / "ome" / "unrelated.ome.tiff.backup").touch()
        (tmp_path / "ome" / "unrelated.tiff").touch()
        (tmp_path / masks_dir / "unrelated.tiff.backup").touch()

        layout = discover_layout(tmp_path, masks_dir=masks_dir)

        assert layout.table == tmp_path / "cells.h5ad"
        assert [sample.name for sample in layout.samples] == ["sample-a", "sample-b"]
        assert not hasattr(layout, "__dict__")
        for sample in layout.samples:
            assert sample.image == tmp_path / "ome" / f"{sample.name}.ome.tiff"
            assert sample.labels == tmp_path / masks_dir / f"{sample.name}.tiff"
            assert not hasattr(sample, "__dict__")

    @pytest.mark.parametrize("kind", ["file", "directory", "escape"])
    def test_optional_image_metadata_artifact(self, tmp_path: Path, kind: str) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _write_paths(root)
        path = root / "images.csv"
        if kind == "directory":
            path.mkdir()
            with pytest.raises(IsADirectoryError):
                discover_layout(root)
        elif kind == "escape":
            outside = tmp_path / "outside.csv"
            outside.touch()
            path.symlink_to(outside)
            with pytest.raises(ReaderFormatError, match="root"):
                discover_layout(root)
        else:
            path.touch()
            assert discover_layout(root).image_metadata == path

    def test_embedded_suffix_and_hyphens_preserved(self, tmp_path: Path) -> None:
        _write_paths(tmp_path)
        name = "sample.tiff-in-tissue"
        (tmp_path / "ome" / "sample-a.ome.tiff").rename(tmp_path / "ome" / f"{name}.ome.tiff")
        (tmp_path / "masks" / "sample-a.tiff").rename(tmp_path / "masks" / f"{name}.tiff")

        layout = discover_layout(tmp_path)

        assert [sample.name for sample in layout.samples] == ["sample-b", name]

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("sample name", "sample_name"),
            ("sample:name", "sample_name"),
            ("sample?", "sample_"),
            ("sample[1]", "sample_1_"),
            ("__private", "_private"),
            (".", "unnamed"),
            ("..", "unnamed"),
        ],
    )
    def test_invalid_sample_names_are_sanitized(self, tmp_path: Path, name: str, expected: str) -> None:
        _write_paths(tmp_path)
        (tmp_path / "ome" / f"{name}.ome.tiff").touch()
        (tmp_path / "masks" / f"{name}.tiff").touch()

        with pytest.warns(UserWarning, match="sample file name") as caught:
            layout = discover_layout(tmp_path)

        assert [str(warning.message) for warning in caught] == [
            (
                f"steinbock: the sample file name {name!r} is not a usable SpatialData element name; using "
                f"{expected!r}. Rename the source files to choose the name explicitly."
            )
        ]
        sample = next(sample for sample in layout.samples if sample.image_name == f"{name}.tiff")
        assert sample.name == expected
        assert sample.image == tmp_path / "ome" / f"{name}.ome.tiff"
        assert sample.labels == tmp_path / "masks" / f"{name}.tiff"

    @pytest.mark.parametrize(("first", "second"), [("sample a", "sample_a"), ("sample a", "SAMPLE:A")])
    def test_names_colliding_after_sanitizing(self, tmp_path: Path, first: str, second: str) -> None:
        _write_paths(tmp_path)
        for name in (first, second):
            (tmp_path / "ome" / f"{name}.ome.tiff").touch()
            (tmp_path / "masks" / f"{name}.tiff").touch()

        with (
            pytest.warns(UserWarning, match="is not a usable SpatialData element name"),
            pytest.raises(ReaderFormatError, match="case-insensitively conflicting sample names") as caught,
        ):
            discover_layout(tmp_path)

        assert repr(sorted([first, second])) in str(caught.value)
        assert caught.value.path == tmp_path

    def test_sanitized_name_colliding_with_a_derived_frame(self, tmp_path: Path) -> None:
        _write_paths(tmp_path)
        (tmp_path / "ome" / "sample-a pixels.ome.tiff").touch()
        (tmp_path / "masks" / "sample-a pixels.tiff").touch()

        with (
            pytest.warns(UserWarning, match="'sample-a_pixels'"),
            pytest.raises(ReaderFormatError, match=re.escape("derived frame names ['sample-a_pixels']")),
        ):
            discover_layout(tmp_path)

    def test_apple_double_companions_are_not_samples(self, tmp_path: Path) -> None:
        _write_paths(tmp_path)
        (tmp_path / "ome" / "._sample-a.ome.tiff").touch()
        (tmp_path / "masks" / "._sample-b.tiff").touch()

        layout = discover_layout(tmp_path)

        assert [sample.name for sample in layout.samples] == ["sample-a", "sample-b"]
        assert [sample.image_name for sample in layout.samples] == ["sample-a.tiff", "sample-b.tiff"]

    @pytest.mark.parametrize("name", ["sample-a_pixels", "SAMPLE-A_Pixels", "sample-b_regionprops_indices"])
    def test_sample_named_like_a_derived_frame(self, tmp_path: Path, name: str) -> None:
        """A sample may not reuse another sample's pixel or centroid frame name, whatever the calibration."""
        _write_paths(tmp_path)
        (tmp_path / "ome" / f"{name}.ome.tiff").touch()
        (tmp_path / "masks" / f"{name}.tiff").touch()

        with pytest.raises(ReaderFormatError, match="derived frame names") as caught:
            discover_layout(tmp_path)

        assert caught.value.path == tmp_path

    def test_derived_suffix_without_base_sample_is_allowed(self, tmp_path: Path) -> None:
        _write_paths(tmp_path)
        (tmp_path / "ome" / "other_pixels.ome.tiff").touch()
        (tmp_path / "masks" / "other_pixels.tiff").touch()

        layout = discover_layout(tmp_path)

        assert [sample.name for sample in layout.samples] == ["other_pixels", "sample-a", "sample-b"]

    def test_case_insensitive_sample_collision(self, tmp_path: Path) -> None:
        _write_paths(tmp_path)
        (tmp_path / "ome" / "SAMPLE-A.ome.tiff").touch()
        (tmp_path / "masks" / "SAMPLE-A.tiff").touch()

        with pytest.raises(ReaderFormatError, match="case-insensitively conflicting"):
            discover_layout(tmp_path)

    @pytest.mark.parametrize("missing", ["cells.h5ad", "ome", "masks"])
    def test_missing_required_artifact(self, tmp_path: Path, missing: str) -> None:
        _write_paths(tmp_path)
        artifact = tmp_path / missing
        if artifact.is_dir():
            for child in artifact.iterdir():
                child.unlink()
            artifact.rmdir()
        else:
            artifact.unlink()

        with pytest.raises(FileNotFoundError) as raised:
            discover_layout(tmp_path)

        assert raised.value.filename == str(artifact)

    @pytest.mark.parametrize("wrong", ["cells.h5ad", "ome", "masks", "ome/sample-a.ome.tiff", "masks/sample-a.tiff"])
    def test_wrong_artifact_kind(self, tmp_path: Path, wrong: str) -> None:
        _write_paths(tmp_path)
        artifact = tmp_path / wrong
        directory = artifact.is_dir()
        if directory:
            for child in artifact.iterdir():
                child.unlink()
            artifact.rmdir()
            artifact.touch()
        else:
            artifact.unlink()
            artifact.mkdir()

        with pytest.raises(NotADirectoryError if directory else IsADirectoryError):
            discover_layout(tmp_path)

    @pytest.mark.parametrize("missing", ["image", "mask", "all"])
    def test_missing_pairs_fail(self, tmp_path: Path, missing: str) -> None:
        _write_paths(tmp_path)
        if missing in {"image", "all"}:
            (tmp_path / "ome" / "sample-a.ome.tiff").unlink()
        if missing in {"mask", "all"}:
            (tmp_path / "masks" / "sample-b.tiff").unlink()
        if missing == "all":
            (tmp_path / "ome" / "sample-b.ome.tiff").unlink()
            (tmp_path / "masks" / "sample-a.tiff").unlink()

        with pytest.raises(ReaderFormatError, match="at least one sample"):
            discover_layout(tmp_path)

    def test_empty_sample_name(self, tmp_path: Path) -> None:
        _write_paths(tmp_path)
        (tmp_path / "ome" / ".ome.tiff").touch()
        (tmp_path / "masks" / ".tiff").touch()

        with pytest.raises(ReaderFormatError, match="nonempty name"):
            discover_layout(tmp_path)

    @pytest.mark.parametrize("relative", ["cells.h5ad", "ome/sample-a.ome.tiff", "masks/sample-a.tiff"])
    def test_artifact_symlink_escape(self, tmp_path: Path, relative: str) -> None:
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        _write_paths(dataset)
        outside = tmp_path / "outside"
        outside.touch()
        artifact = dataset / relative
        artifact.unlink()
        artifact.symlink_to(outside)

        with pytest.raises(ReaderFormatError, match="outside"):
            discover_layout(dataset)

    @pytest.mark.parametrize("directory", ["ome", "masks"])
    def test_directory_symlink_escape(self, tmp_path: Path, directory: str) -> None:
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        _write_paths(dataset)
        outside = tmp_path / "outside"
        artifact = dataset / directory
        artifact.rename(outside)
        artifact.symlink_to(outside, target_is_directory=True)

        with pytest.raises(ReaderFormatError, match="outside"):
            discover_layout(dataset)

    def test_root_not_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "not-a-directory"
        root.touch()

        with pytest.raises(NotADirectoryError):
            discover_layout(root)


class TestDiscoverLayoutSources:
    """Resolve the mask directory and the table file under the root's path policy."""

    def test_nested_masks_dir_and_relative_table_file(self, tmp_path: Path) -> None:
        _write_paths(tmp_path)
        (tmp_path / "segmentation").mkdir()
        (tmp_path / "masks").rename(tmp_path / "segmentation" / "deepcell")
        (tmp_path / "exports").mkdir()
        (tmp_path / "cells.h5ad").rename(tmp_path / "exports" / "objects.h5ad")

        layout = discover_layout(tmp_path, masks_dir="segmentation/deepcell", table_file="exports/objects.h5ad")

        assert layout.table == tmp_path / "exports" / "objects.h5ad"
        assert [sample.labels for sample in layout.samples] == [
            tmp_path / "segmentation" / "deepcell" / "sample-a.tiff",
            tmp_path / "segmentation" / "deepcell" / "sample-b.tiff",
        ]

    def test_absolute_table_file_outside_root(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _write_paths(root)
        table = tmp_path / "objects.h5ad"
        (root / "cells.h5ad").rename(table)

        assert discover_layout(root, table_file=table).table == table

    @pytest.mark.parametrize("policy", ["strict", "relaxed"])
    def test_symlinked_table_file(self, tmp_path: Path, policy: PathPolicy) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _write_paths(root)
        target = tmp_path / "object-store" / "0123abcd"
        target.parent.mkdir()
        (root / "cells.h5ad").rename(target)
        (root / "objects.h5ad").symlink_to(target)

        if policy == "strict":
            with pytest.raises(ReaderFormatError, match="path_policy='relaxed'") as caught:
                discover_layout(root, table_file="objects.h5ad", path_policy=policy)
            assert caught.value.context.artifact == "cell table"
        else:
            layout = discover_layout(root, table_file="objects.h5ad", path_policy=policy)
            assert layout.table == root / "objects.h5ad"

    @pytest.mark.parametrize("masks_dir", ["absent", "masks/sample-a.tiff"])
    def test_masks_dir_must_be_an_existing_directory(self, tmp_path: Path, masks_dir: str) -> None:
        _write_paths(tmp_path)

        with pytest.raises(FileNotFoundError if masks_dir == "absent" else NotADirectoryError):
            discover_layout(tmp_path, masks_dir=masks_dir)


class TestValidateMasksDir:
    """Accept relative directory paths that cannot leave the dataset root."""

    @pytest.mark.parametrize("masks_dir", ["masks", "masks_deepcell", "segmentation/deepcell", "./masks"])
    def test_relative_directories(self, masks_dir: str) -> None:
        assert validate_masks_dir(masks_dir) == masks_dir

    @pytest.mark.parametrize("masks_dir", ["", ".", "/data/masks", "..", "../masks", "masks/../../other"])
    def test_absolute_empty_or_escaping_paths(self, masks_dir: str) -> None:
        with pytest.raises(ValueError, match="masks_dir must be a relative directory inside the dataset"):
            validate_masks_dir(masks_dir)

    @pytest.mark.parametrize("masks_dir", [None, 1, b"masks", Path("masks")])
    def test_non_string(self, masks_dir: object) -> None:
        with pytest.raises(TypeError, match=f"masks_dir must be a string, not {type(masks_dir).__name__}"):
            validate_masks_dir(masks_dir)


class TestValidateTableFile:
    """Accept strings and paths and return a path."""

    @pytest.mark.parametrize("table_file", ["objects.h5ad", Path("exports/objects.h5ad")])
    def test_paths(self, table_file: str | Path) -> None:
        assert validate_table_file(table_file) == Path(table_file)

    @pytest.mark.parametrize("table_file", [None, 1, b"cells.h5ad"])
    def test_non_path(self, table_file: object) -> None:
        with pytest.raises(TypeError, match=f"table_file must be a str or Path, not {type(table_file).__name__}"):
            validate_table_file(table_file)
