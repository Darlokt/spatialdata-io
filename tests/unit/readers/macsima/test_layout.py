"""Deterministic discovery of single- and multiple-folder MACSima layouts."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.macsima._layout import discover_layout, validate_layout_options

if TYPE_CHECKING:
    from pathlib import Path


def _touch(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


class TestValidateLayoutOptions:
    """Validate the parsing style and exact exclusions before discovery."""

    def test_returns_owned_names(self) -> None:
        names = ["a", "b"]

        assert validate_layout_options("processed_multiple_folders", names) == frozenset({"a", "b"})

    def test_single_folder_has_no_exclusions(self) -> None:
        assert validate_layout_options("processed_single_folder", ()) == frozenset()

    @pytest.mark.parametrize(
        ("style", "names", "error", "match"),
        [
            ("raw", (), ValueError, "parsing_style must be one of"),
            (1, (), TypeError, "parsing_style must be a string; found int"),
            ("processed_multiple_folders", "run", TypeError, "sequence of folder names"),
            ("processed_multiple_folders", ("",), ValueError, "must not contain empty names"),
            ("processed_multiple_folders", (1,), TypeError, "only strings"),
            ("processed_single_folder", ("run",), ValueError, "requires parsing_style='processed_multiple_folders'"),
        ],
    )
    def test_rejects(self, style: str, names: object, error: type[Exception], match: str) -> None:
        with pytest.raises(error, match=match):
            validate_layout_options(style, names)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.


class TestDiscoverSingleFolder:
    """Stack every visible TIFF below the root in lexical path order."""

    def test_finds_tiffs_recursively_and_ignores_other_files(self, tmp_path: Path) -> None:
        root = tmp_path / "my run"
        _touch(
            root / "b.tif",
            root / "a.OME.TIFF",
            root / "nested" / "c.tiff",
            root / "notes.txt",
            root / "overview.png",
            root / ".hidden.tif",
            root / ".cache" / "d.tif",
            root / "__MACOSX" / "._b.tif",
        )
        (root / "folder.tif").mkdir()
        (root / "linked_folder.tif").symlink_to(root / "nested", target_is_directory=True)

        with pytest.warns(UserWarning, match=r"stack folder name 'my run' .*; using 'my_run'"):
            layout = discover_layout(root, parsing_style="processed_single_folder", exclude_folder_names=frozenset())

        (stack,) = layout.stacks
        assert stack.name == "my_run"
        assert stack.directory == root.resolve()
        assert [path.relative_to(root.resolve()).as_posix() for path in stack.files] == [
            "a.OME.TIFF",
            "b.tif",
            "nested/c.tiff",
        ]

    @pytest.mark.parametrize("name", ["gone.tif", "nested/gone.TIFF"])
    def test_dangling_tiff_symlink_fails(self, tmp_path: Path, name: str) -> None:
        _touch(tmp_path / "run" / "a.tif")
        link = tmp_path / "run" / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(tmp_path / "missing.tif")

        with pytest.raises(FileNotFoundError, match="dangling symlink") as caught:
            discover_layout(tmp_path / "run", parsing_style="processed_single_folder", exclude_folder_names=frozenset())
        assert caught.value.filename == str(link)

    def test_ignored_dangling_symlinks_do_not_fail(self, tmp_path: Path) -> None:
        _touch(tmp_path / "run" / "a.tif")
        for name in (".hidden.tif", "notes.txt"):
            (tmp_path / "run" / name).symlink_to(tmp_path / "missing")

        layout = discover_layout(
            tmp_path / "run", parsing_style="processed_single_folder", exclude_folder_names=frozenset()
        )

        assert [path.name for path in layout.stacks[0].files] == ["a.tif"]

    def test_non_regular_tiff_fails(self, tmp_path: Path) -> None:
        _touch(tmp_path / "run" / "a.tif")
        os.mkfifo(tmp_path / "run" / "pipe.tif")

        with pytest.raises(ReaderFormatError, match="non-regular"):
            discover_layout(tmp_path / "run", parsing_style="processed_single_folder", exclude_folder_names=frozenset())

    def test_rejects_a_folder_without_tiffs(self, tmp_path: Path) -> None:
        _touch(tmp_path / "notes.txt")

        with pytest.raises(ReaderFormatError, match=r"no \.tif or \.tiff files"):
            discover_layout(tmp_path, parsing_style="processed_single_folder", exclude_folder_names=frozenset())

    def test_missing_and_wrong_kind_roots(self, tmp_path: Path) -> None:
        _touch(tmp_path / "file.tif")

        with pytest.raises(FileNotFoundError):
            discover_layout(
                tmp_path / "missing", parsing_style="processed_single_folder", exclude_folder_names=frozenset()
            )
        with pytest.raises(NotADirectoryError):
            discover_layout(
                tmp_path / "file.tif", parsing_style="processed_single_folder", exclude_folder_names=frozenset()
            )


class TestDiscoverMultipleFolders:
    """Make one stack per visible immediate subfolder in sorted order."""

    def test_one_stack_per_subfolder(self, tmp_path: Path) -> None:
        _touch(
            tmp_path / "run_2" / "x.tif",
            tmp_path / "run" / "nested" / "y.tif",
            tmp_path / "run" / "x.tif",
            tmp_path / "root-level.tif",
            tmp_path / ".hidden" / "x.tif",
        )

        layout = discover_layout(tmp_path, parsing_style="processed_multiple_folders", exclude_folder_names=frozenset())

        assert [stack.name for stack in layout.stacks] == ["run", "run_2"]
        assert [stack.directory for stack in layout.stacks] == [
            (tmp_path / "run").resolve(),
            (tmp_path / "run_2").resolve(),
        ]
        assert [path.name for path in layout.stacks[0].files] == ["y.tif", "x.tif"]

    def test_exclusion_matches_exact_names_only(self, tmp_path: Path) -> None:
        _touch(tmp_path / "run" / "x.tif", tmp_path / "run_2" / "x.tif")

        layout = discover_layout(
            tmp_path, parsing_style="processed_multiple_folders", exclude_folder_names=frozenset({"run"})
        )

        assert [stack.name for stack in layout.stacks] == ["run_2"]

    def test_rejects_exclusions_that_match_no_folder(self, tmp_path: Path) -> None:
        _touch(tmp_path / "run" / "x.tif")

        with pytest.raises(ValueError, match=r"\['ru'\] match no folder"):
            discover_layout(
                tmp_path, parsing_style="processed_multiple_folders", exclude_folder_names=frozenset({"ru"})
            )

    def test_skips_folders_without_tiffs_with_a_warning(self, tmp_path: Path) -> None:
        _touch(tmp_path / "run" / "x.tif", tmp_path / "empty" / "notes.txt")

        with pytest.warns(UserWarning, match="No TIFF files found in .*empty.*skipping"):
            layout = discover_layout(
                tmp_path, parsing_style="processed_multiple_folders", exclude_folder_names=frozenset()
            )

        assert [stack.name for stack in layout.stacks] == ["run"]

    def test_rejects_when_no_folder_has_tiffs(self, tmp_path: Path) -> None:
        _touch(tmp_path / "run" / "x.tif", tmp_path / "root.tif")

        with pytest.raises(ReaderFormatError, match="no included subfolder"):
            discover_layout(
                tmp_path, parsing_style="processed_multiple_folders", exclude_folder_names=frozenset({"run"})
            )

    def test_repairs_invalid_folder_names_with_a_warning(self, tmp_path: Path) -> None:
        _touch(tmp_path / "first run" / "x.tif", tmp_path / "second" / "x.tif")

        with pytest.warns(UserWarning, match="first run") as caught:
            layout = discover_layout(
                tmp_path, parsing_style="processed_multiple_folders", exclude_folder_names=frozenset()
            )

        assert [str(warning.message) for warning in caught] == [
            (
                "macsima: the stack folder name 'first run' is not a usable SpatialData element name; "
                "using 'first_run'. Rename the source files to choose the name explicitly."
            )
        ]
        assert [stack.name for stack in layout.stacks] == ["first_run", "second"]
        assert layout.stacks[0].directory == (tmp_path / "first run").resolve()

    def test_symlinked_folder_keeps_its_own_name(self, tmp_path: Path) -> None:
        # Under the strict policy an in-root alias resolves to its target folder.
        _touch(tmp_path / "data" / "x.tif")
        (tmp_path / "linked").symlink_to(tmp_path / "data", target_is_directory=True)

        layout = discover_layout(tmp_path, parsing_style="processed_multiple_folders", exclude_folder_names=frozenset())

        assert [stack.name for stack in layout.stacks] == ["data", "linked"]
        assert [stack.directory for stack in layout.stacks] == [(tmp_path / "data").resolve()] * 2

    def test_rejects_names_that_collide_after_sanitizing(self, tmp_path: Path) -> None:
        _touch(tmp_path / "a b" / "x.tif", tmp_path / "a_b" / "x.tif")

        with (
            pytest.warns(UserWarning, match="using 'a_b'"),
            pytest.raises(ReaderFormatError, match=r"folders \['a b', 'a_b'\] whose element names are equal"),
        ):
            discover_layout(tmp_path, parsing_style="processed_multiple_folders", exclude_folder_names=frozenset())

    def test_rejects_names_that_differ_only_in_case(self, tmp_path: Path) -> None:
        _touch(tmp_path / "Run" / "x.tif", tmp_path / "run" / "x.tif")
        if len(list(tmp_path.iterdir())) == 1:
            pytest.skip("The temporary filesystem is case-insensitive.")

        with pytest.raises(ReaderFormatError, match=r"folders \['Run', 'run'\] whose element names are equal ignoring"):
            discover_layout(tmp_path, parsing_style="processed_multiple_folders", exclude_folder_names=frozenset())
