"""Tests for deterministic reader filesystem primitives."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import (
    DatasetRoot,
    _core,
    normalize_dataset_root,
    optional_directory,
    optional_file,
    optional_unique_path,
    require_caller_directory,
    require_caller_file,
    require_directory,
    require_file,
    require_metadata_directory,
    require_metadata_file,
    require_unique_path,
    sorted_paths,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_mock import MockerFixture

_CONTEXT = ReaderErrorContext(reader="test_reader", artifact="test_artifact", detected_version="2")


def _root(path: Path) -> DatasetRoot:
    return normalize_dataset_root(path, context=_CONTEXT)


class _OneShotPaths:
    def __init__(self, values: tuple[Path, ...], *, error: OSError | None = None) -> None:
        self._values = values
        self._error = error
        self.iterations = 0

    def __iter__(self) -> Iterator[Path]:
        self.iterations += 1
        if self.iterations > 1:
            message = "Iterable was consumed more than once."
            raise RuntimeError(message)
        yield from self._values
        if self._error is not None:
            raise self._error


class TestNormalizeDatasetRoot:
    def test_returns_canonical_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(target, target_is_directory=True)

        result = normalize_dataset_root(alias, context=_CONTEXT)

        assert result == target.resolve()
        assert Path(result).is_absolute()

    def test_expands_user_path(self, tmp_path: Path, mocker: MockerFixture) -> None:
        root = tmp_path / "root"
        root.mkdir()
        expanduser = mocker.patch.object(Path, "expanduser", autospec=True, return_value=root)

        assert normalize_dataset_root("~/root", context=_CONTEXT) == root.resolve()
        expanduser.assert_called_once()

    def test_rejects_missing_root_with_standard_fields(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"

        with pytest.raises(FileNotFoundError) as raised:
            normalize_dataset_root(missing, context=_CONTEXT)

        assert raised.value.errno == errno.ENOENT
        assert raised.value.filename == str(missing)
        assert isinstance(raised.value.__cause__, FileNotFoundError)

    def test_rejects_file_root(self, tmp_path: Path) -> None:
        path = tmp_path / "file"
        path.touch()

        with pytest.raises(NotADirectoryError) as raised:
            normalize_dataset_root(path, context=_CONTEXT)

        assert raised.value.errno == errno.ENOTDIR
        assert raised.value.filename == str(path)

    def test_rejects_unsupported_root(self, tmp_path: Path) -> None:
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)

        with pytest.raises(ReaderFormatError, match="non-regular"):
            normalize_dataset_root(fifo, context=_CONTEXT)


class TestRequireFile:
    def test_returns_canonical_target(self, tmp_path: Path) -> None:
        target = tmp_path / "target.data"
        target.touch()
        alias = tmp_path / "artifact.csv"
        alias.symlink_to(target)

        assert require_file(alias, context=_CONTEXT) == target.resolve()

    def test_rejects_missing_and_dangling_paths(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="missing path component"):
            require_file(tmp_path / "missing", context=_CONTEXT)

        dangling = tmp_path / "dangling"
        dangling.symlink_to(tmp_path / "missing")
        with pytest.raises(FileNotFoundError, match="dangling symlink"):
            require_file(dangling, context=_CONTEXT)

    def test_rejects_directory_and_unsupported_kind(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError) as raised:
            require_file(tmp_path, context=_CONTEXT)
        assert raised.value.errno == errno.EISDIR
        assert raised.value.filename == str(tmp_path)

        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        with pytest.raises(ReaderFormatError, match="non-regular"):
            require_file(fifo, context=_CONTEXT)

    def test_preserves_resolution_error(self, tmp_path: Path, mocker: MockerFixture) -> None:
        error = RuntimeError("symlink loop")
        mocker.patch.object(Path, "resolve", side_effect=error)

        with pytest.raises(RuntimeError) as raised:
            require_file(tmp_path / "file", context=_CONTEXT)

        assert raised.value is error
        assert "test_reader" in raised.value.__notes__[0]

    def test_contextualizes_stat_error(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "file"
        path.touch()
        error = PermissionError("denied")
        mocker.patch.object(Path, "stat", side_effect=error)

        with pytest.raises(PermissionError) as raised:
            require_file(path, context=_CONTEXT)

        assert raised.value is error
        assert "test_reader" in raised.value.__notes__[0]

    def test_reports_target_disappearing_during_kind_check(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "file"
        path.touch()
        mocker.patch.object(_core, "_resolve_existing", return_value=path)
        path.unlink()

        with pytest.raises(FileNotFoundError, match="disappeared"):
            require_file(path, context=_CONTEXT)


class TestRequireDirectory:
    def test_returns_canonical_target(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(target, target_is_directory=True)

        assert require_directory(alias, context=_CONTEXT) == target.resolve()

    def test_rejects_file_missing_and_unsupported_kind(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file"
        file_path.touch()
        with pytest.raises(NotADirectoryError):
            require_directory(file_path, context=_CONTEXT)
        with pytest.raises(FileNotFoundError):
            require_directory(tmp_path / "missing", context=_CONTEXT)

        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        with pytest.raises(ReaderFormatError):
            require_directory(fifo, context=_CONTEXT)


class TestOptionalFile:
    def test_returns_none_only_for_absent_leaf_with_valid_parent(self, tmp_path: Path) -> None:
        assert optional_file(tmp_path / "missing", context=_CONTEXT) is None

    def test_returns_canonical_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.touch()
        alias = tmp_path / "alias"
        alias.symlink_to(target)

        assert optional_file(alias, context=_CONTEXT) == target.resolve()

    @pytest.mark.parametrize("parent_state", ["missing", "dangling"])
    def test_rejects_missing_or_dangling_parent(self, tmp_path: Path, parent_state: str) -> None:
        parent = tmp_path / "parent"
        if parent_state == "dangling":
            parent.symlink_to(tmp_path / "missing", target_is_directory=True)

        with pytest.raises(FileNotFoundError, match="parent directory"):
            optional_file(parent / "artifact", context=_CONTEXT)

    def test_rejects_dangling_leaf_and_wrong_kind(self, tmp_path: Path) -> None:
        dangling = tmp_path / "dangling"
        dangling.symlink_to(tmp_path / "missing")
        with pytest.raises(FileNotFoundError, match="dangling symlink"):
            optional_file(dangling, context=_CONTEXT)
        with pytest.raises(IsADirectoryError):
            optional_file(tmp_path, context=_CONTEXT)

    def test_contextualizes_lstat_error(self, tmp_path: Path, mocker: MockerFixture) -> None:
        error = PermissionError("denied")
        mocker.patch.object(Path, "lstat", side_effect=error)

        with pytest.raises(PermissionError) as raised:
            optional_file(tmp_path / "file", context=_CONTEXT)

        assert raised.value is error
        assert "test_reader" in raised.value.__notes__[0]


class TestOptionalDirectory:
    def test_returns_none_for_absent_leaf(self, tmp_path: Path) -> None:
        assert optional_directory(tmp_path / "missing", context=_CONTEXT) is None

    def test_returns_canonical_existing_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(target, target_is_directory=True)

        assert optional_directory(alias, context=_CONTEXT) == target.resolve()

    def test_rejects_missing_parent_file_and_dangling_leaf(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="parent directory"):
            optional_directory(tmp_path / "missing" / "directory", context=_CONTEXT)

        file_path = tmp_path / "file"
        file_path.touch()
        with pytest.raises(NotADirectoryError):
            optional_directory(file_path, context=_CONTEXT)

        dangling = tmp_path / "dangling"
        dangling.symlink_to(tmp_path / "missing", target_is_directory=True)
        with pytest.raises(FileNotFoundError):
            optional_directory(dangling, context=_CONTEXT)


class TestRequireCallerFile:
    def test_accepts_relative_internal_and_absolute_external_files(self, tmp_path: Path) -> None:
        root_path = tmp_path / "root"
        root_path.mkdir()
        internal = root_path / "internal"
        internal.touch()
        external = tmp_path / "external"
        external.touch()
        root = _root(root_path)

        assert require_caller_file(root, "internal", context=_CONTEXT) == internal.resolve()
        assert require_caller_file(root, external, context=_CONTEXT) == external.resolve()

    def test_expands_user_path(self, tmp_path: Path, mocker: MockerFixture) -> None:
        root_path = tmp_path / "root"
        root_path.mkdir()
        external = tmp_path / "external"
        external.touch()
        root = _root(root_path)
        expanduser = mocker.patch.object(Path, "expanduser", autospec=True, return_value=external)

        assert require_caller_file(root, "~/external", context=_CONTEXT) == external.resolve()
        expanduser.assert_called_once()

    def test_rejects_relative_traversal_and_symlink_escape(self, tmp_path: Path) -> None:
        root_path = tmp_path / "root"
        root_path.mkdir()
        external = tmp_path / "external"
        external.touch()
        (root_path / "alias").symlink_to(external)
        root = _root(root_path)

        with pytest.raises(ReaderFormatError, match="outside the dataset root"):
            require_caller_file(root, Path("..") / "external", context=_CONTEXT)
        with pytest.raises(ReaderFormatError, match="outside the dataset root"):
            require_caller_file(root, "alias", context=_CONTEXT)

    def test_rejects_missing_relative_traversal_before_resolution(self, tmp_path: Path, mocker: MockerFixture) -> None:
        root_path = tmp_path / "root"
        root_path.mkdir()
        root = _root(root_path)
        resolve = mocker.spy(_core, "_resolve_existing")

        with pytest.raises(ReaderFormatError, match="lexically outside the dataset root"):
            require_caller_file(root, Path("..") / "missing", context=_CONTEXT)

        resolve.assert_not_called()

    def test_rejects_directory(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError):
            require_caller_file(_root(tmp_path), tmp_path, context=_CONTEXT)

    def test_resolves_and_checks_kind_once(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "file"
        path.touch()
        root = _root(tmp_path)
        resolve = mocker.spy(_core, "_resolve_existing")
        inspect_kind = mocker.spy(_core, "_filesystem_kind")

        require_caller_file(root, "file", context=_CONTEXT)

        resolve.assert_called_once()
        inspect_kind.assert_called_once()


class TestRequireCallerDirectory:
    def test_accepts_internal_and_external_directories(self, tmp_path: Path) -> None:
        root_path = tmp_path / "root"
        root_path.mkdir()
        internal = root_path / "internal"
        internal.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        root = _root(root_path)

        assert require_caller_directory(root, "internal", context=_CONTEXT) == internal.resolve()
        assert require_caller_directory(root, external, context=_CONTEXT) == external.resolve()

    def test_rejects_file_and_relative_escape(self, tmp_path: Path) -> None:
        root_path = tmp_path / "root"
        root_path.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        file_path = root_path / "file"
        file_path.touch()
        root = _root(root_path)

        with pytest.raises(NotADirectoryError):
            require_caller_directory(root, "file", context=_CONTEXT)
        with pytest.raises(ReaderFormatError):
            require_caller_directory(root, Path("..") / "external", context=_CONTEXT)


class TestRequireMetadataFile:
    def test_accepts_relative_and_absolute_internal_files(self, tmp_path: Path) -> None:
        path = tmp_path / "artifact"
        path.touch()
        root = _root(tmp_path)

        assert require_metadata_file(root, "artifact", context=_CONTEXT) == path.resolve()
        assert require_metadata_file(root, path, context=_CONTEXT) == path.resolve()

    def test_rejects_external_and_symlink_escape(self, tmp_path: Path) -> None:
        root_path = tmp_path / "root"
        root_path.mkdir()
        external = tmp_path / "external"
        external.touch()
        (root_path / "alias").symlink_to(external)
        root = _root(root_path)

        with pytest.raises(ReaderFormatError):
            require_metadata_file(root, external, context=_CONTEXT)
        with pytest.raises(ReaderFormatError):
            require_metadata_file(root, "alias", context=_CONTEXT)

    def test_rejects_missing_external_path_before_resolution(self, tmp_path: Path, mocker: MockerFixture) -> None:
        root_path = tmp_path / "root"
        root_path.mkdir()
        root = _root(root_path)
        resolve = mocker.spy(_core, "_resolve_existing")

        with pytest.raises(ReaderFormatError, match="lexically outside the dataset root"):
            require_metadata_file(root, tmp_path / "missing", context=_CONTEXT)

        resolve.assert_not_called()

    def test_treats_tilde_literally(self, tmp_path: Path, mocker: MockerFixture) -> None:
        tilde_directory = tmp_path / "~vendor"
        tilde_directory.mkdir()
        artifact = tilde_directory / "artifact"
        artifact.touch()
        root = _root(tmp_path)
        expanduser = mocker.patch.object(Path, "expanduser", autospec=True)

        assert require_metadata_file(root, "~vendor/artifact", context=_CONTEXT) == artifact.resolve()
        expanduser.assert_not_called()

    def test_rejects_directory(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError):
            require_metadata_file(_root(tmp_path), tmp_path, context=_CONTEXT)


class TestRequireMetadataDirectory:
    def test_accepts_internal_directory(self, tmp_path: Path) -> None:
        directory = tmp_path / "store.zarr"
        directory.mkdir()

        assert require_metadata_directory(_root(tmp_path), "store.zarr", context=_CONTEXT) == directory.resolve()

    def test_rejects_external_directory_and_file(self, tmp_path: Path) -> None:
        root_path = tmp_path / "root"
        root_path.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        file_path = root_path / "file"
        file_path.touch()
        root = _root(root_path)

        with pytest.raises(ReaderFormatError):
            require_metadata_directory(root, external, context=_CONTEXT)
        with pytest.raises(NotADirectoryError):
            require_metadata_directory(root, "file", context=_CONTEXT)


class TestSortedPaths:
    def test_sorts_without_resolving_or_deduplicating(self, tmp_path: Path) -> None:
        values = _OneShotPaths((Path("z/missing"), Path("a/missing"), Path("a/missing")))

        result = sorted_paths(values, context=_CONTEXT, search_path=tmp_path)

        assert result == (Path("a/missing"), Path("a/missing"), Path("z/missing"))
        assert values.iterations == 1

    def test_contextualizes_iteration_error(self, tmp_path: Path) -> None:
        error = PermissionError("scan denied")
        values = _OneShotPaths((Path("a"),), error=error)

        with pytest.raises(PermissionError) as raised:
            sorted_paths(values, context=_CONTEXT, search_path=tmp_path)

        assert raised.value is error
        assert str(tmp_path) in raised.value.__notes__[0]


class TestRequireUniquePath:
    def test_rejects_zero_candidates_with_search_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError) as raised:
            require_unique_path((), context=_CONTEXT, search_path=tmp_path)

        assert raised.value.filename == str(tmp_path)
        assert "exactly one candidate" in str(raised.value)

    def test_returns_sole_candidate_without_sorting(self, tmp_path: Path, mocker: MockerFixture) -> None:
        candidate = Path("missing.csv")
        ordered = mocker.patch("builtins.sorted")

        assert require_unique_path((candidate,), context=_CONTEXT, search_path=tmp_path) is candidate
        ordered.assert_not_called()

    def test_reports_all_conflicts_in_order_and_keeps_aliases(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.touch()
        first = tmp_path / "z"
        second = tmp_path / "a"
        first.symlink_to(target)
        second.symlink_to(target)

        with pytest.raises(ReaderFormatError) as raised:
            require_unique_path((first, second), context=_CONTEXT, search_path=tmp_path)

        assert raised.value.path == tmp_path
        assert str(raised.value).index(str(second)) < str(raised.value).index(str(first))
        assert raised.value.found.startswith("2 candidates")

    def test_consumes_iterable_once_and_contextualizes_errors(self, tmp_path: Path) -> None:
        values = _OneShotPaths((Path("only"),))
        assert require_unique_path(values, context=_CONTEXT, search_path=tmp_path) == Path("only")
        assert values.iterations == 1

        error = PermissionError("scan denied")
        with pytest.raises(PermissionError) as raised:
            require_unique_path(_OneShotPaths((), error=error), context=_CONTEXT, search_path=tmp_path)
        assert raised.value is error
        assert raised.value.__notes__


class TestOptionalUniquePath:
    def test_returns_none_or_sole_candidate(self, tmp_path: Path) -> None:
        assert optional_unique_path((), context=_CONTEXT, search_path=tmp_path) is None
        candidate = Path("candidate")
        assert optional_unique_path((candidate,), context=_CONTEXT, search_path=tmp_path) is candidate

    def test_reports_ordered_conflicts_and_iteration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError) as raised:
            optional_unique_path((Path("z"), Path("a")), context=_CONTEXT, search_path=tmp_path)
        assert raised.value.expected == "zero or one candidate"
        assert str(raised.value).index("'a'") < str(raised.value).index("'z'")

        error = PermissionError("scan denied")
        with pytest.raises(PermissionError) as iteration:
            optional_unique_path(_OneShotPaths((), error=error), context=_CONTEXT, search_path=tmp_path)
        assert iteration.value.__notes__
