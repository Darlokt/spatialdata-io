from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pooch
import pytest
import requests

if TYPE_CHECKING:
    from types import ModuleType


SCRIPT_DIR = Path(__file__).parents[2] / "scripts" / "test_data_downloader"
SHA256 = "sha256:" + "0" * 64


def _load_module(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


manifest = _load_module("manifest", SCRIPT_DIR / "manifest.py")
download_test_data = _load_module("downloader", SCRIPT_DIR / "downloader.py")


def _make_dataset(
    key: str = "example",
    *,
    group: str = "group",
    url: str | None = None,
    extracted_dir: str | None = None,
    source: str = "example",
    test_path: str = "",
    doi: str = "",
) -> Any:
    return download_test_data.TestDataset(
        key=key,
        group=group,
        extracted_dir=extracted_dir or key,
        source=source,
        test_path=test_path,
        url="" if doi else url or f"https://example.com/{key}.zip",
        known_hash="" if doi else SHA256,
        doi=doi,
    )


class TestFetchDataset:
    def test_uses_dataset_manifest_module(self) -> None:
        assert download_test_data.TestDataset.__module__ == "manifest"
        assert all(isinstance(dataset, manifest.TestDataset) for dataset in download_test_data.DATASETS)

    def test_fetches_archive_with_pooch_registry_and_unzip(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        dataset = _make_dataset()
        seen_create: dict[str, Any] = {}
        seen_fetch: dict[str, Any] = {}

        class Manager:
            def fetch(self, file_name: str, processor: Any = None) -> None:
                seen_fetch.update(file_name=file_name, processor=processor)

        def create(**kwargs: Any) -> Manager:
            seen_create.update(kwargs)
            return Manager()

        monkeypatch.setattr(download_test_data.pooch, "create", create)

        download_test_data._fetch_dataset(dataset, tmp_path, tmp_path / dataset.extracted_dir)

        archive_name = f"{dataset.key}.zip"
        assert seen_create == {
            "path": tmp_path,
            "base_url": "",
            "registry": {archive_name: dataset.known_hash},
            "urls": {archive_name: dataset.url},
            "retry_if_failed": download_test_data.DOWNLOAD_RETRIES,
        }
        assert seen_fetch["file_name"] == archive_name
        assert isinstance(seen_fetch["processor"], pooch.Unzip)
        assert seen_fetch["processor"].extract_dir == dataset.extracted_dir

    def test_loads_hashes_and_fetches_all_files_from_doi_registry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        dataset = _make_dataset("doi", doi="10.5281/zenodo.123")
        fetched: list[str] = []
        registry_loaded = False

        class Manager:
            registry: dict[str, str] = {}

            @property
            def registry_files(self) -> list[str]:
                return list(self.registry)

            def load_registry_from_doi(self) -> None:
                nonlocal registry_loaded
                registry_loaded = True
                self.registry = {
                    "first.tif": "md5:" + "0" * 32,
                    "second.tif": "md5:" + "1" * 32,
                }

            def fetch(self, file_name: str) -> None:
                assert self.registry[file_name].startswith("md5:")
                fetched.append(file_name)

        def create(**kwargs: Any) -> Manager:
            assert kwargs == {
                "path": tmp_path / dataset.extracted_dir,
                "base_url": "doi:10.5281/zenodo.123/",
                "registry": {},
                "retry_if_failed": download_test_data.DOWNLOAD_RETRIES,
            }
            return Manager()

        monkeypatch.setattr(download_test_data.pooch, "create", create)

        download_test_data._fetch_dataset(dataset, tmp_path, tmp_path / dataset.extracted_dir)

        assert registry_loaded
        assert fetched == ["first.tif", "second.tif"]

    def test_rejects_empty_doi_registry(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        dataset = _make_dataset("doi", doi="10.5281/zenodo.123")

        class Manager:
            registry_files: list[str] = []

            def load_registry_from_doi(self) -> None:
                return None

        monkeypatch.setattr(download_test_data.pooch, "create", lambda **kwargs: Manager())

        with pytest.raises(ValueError, match="contains no files"):
            download_test_data._fetch_dataset(dataset, tmp_path, tmp_path / dataset.extracted_dir)


class TestDatasetManifest:
    def test_manifest_is_valid(self) -> None:
        manifest.validate_datasets()

    def test_returns_dataset_by_key(self) -> None:
        dataset = manifest.get_dataset("seqfish")

        assert dataset.extracted_dir == "seqfish-2-test-dataset"
        assert dataset.known_hash.startswith("sha256:")

    def test_reports_unknown_dataset_key(self) -> None:
        with pytest.raises(KeyError, match="Unknown test dataset key 'missing'"):
            manifest.get_dataset("missing")

    def test_can_filter_datasets_by_group(self) -> None:
        datasets = manifest.datasets_by_group("macsima")

        assert {dataset.key for dataset in datasets} == {"macsima_omap10", "macsima_omap23"}
        assert all(dataset.doi for dataset in datasets)

    def test_loads_archive_dataset_from_toml(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text(
            f"""
[[datasets]]
key = "example"
group = "group"
extracted_dir = "example"
source = "example source"
test_path = "nested"
url = "https://example.com/example.zip"
known_hash = "{SHA256}"
""",
            encoding="utf-8",
        )

        datasets = manifest.load_datasets(manifest_path)

        assert datasets == (
            manifest.TestDataset(
                key="example",
                group="group",
                extracted_dir="example",
                source="example source",
                test_path="nested",
                url="https://example.com/example.zip",
                known_hash=SHA256,
            ),
        )

    def test_loads_doi_dataset_from_toml(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text(
            """
[[datasets]]
key = "example"
group = "group"
extracted_dir = "example"
source = "example source"
doi = "10.5281/zenodo.123"
""",
            encoding="utf-8",
        )

        (dataset,) = manifest.load_datasets(manifest_path)

        assert dataset.url == ""
        assert dataset.known_hash == ""
        assert dataset.doi == "10.5281/zenodo.123"

    def test_rejects_duplicate_dataset_keys(self) -> None:
        first = _make_dataset("duplicate", extracted_dir="first")
        second = _make_dataset("duplicate", extracted_dir="second")

        with pytest.raises(ValueError, match="Duplicate test dataset key: 'duplicate'"):
            manifest.validate_datasets((first, second))

    def test_rejects_duplicate_extracted_dirs(self) -> None:
        first = _make_dataset("first", extracted_dir="duplicate")
        second = _make_dataset("second", extracted_dir="duplicate")

        with pytest.raises(ValueError, match="Duplicate test dataset extracted_dir: 'duplicate'"):
            manifest.validate_datasets((first, second))

    def test_rejects_empty_required_manifest_fields(self) -> None:
        dataset = _make_dataset("missing-group", group="")

        with pytest.raises(ValueError, match="Dataset 'missing-group' has empty group"):
            manifest.validate_datasets((dataset,))

    def test_rejects_test_path_outside_extracted_dir(self) -> None:
        dataset = _make_dataset("unsafe-test-path", test_path="../outside")

        with pytest.raises(ValueError, match="test_path must be a relative path"):
            manifest.validate_datasets((dataset,))

    def test_rejects_invalid_known_hash(self) -> None:
        dataset = _make_dataset("invalid-hash")
        dataset = manifest.TestDataset(
            key=dataset.key,
            group=dataset.group,
            extracted_dir=dataset.extracted_dir,
            source=dataset.source,
            url=dataset.url,
            known_hash="sha256:not-a-hash",
        )

        with pytest.raises(ValueError, match="known_hash"):
            manifest.validate_datasets((dataset,))

    @pytest.mark.parametrize(
        ("url", "known_hash", "doi"),
        [
            ("", "", ""),
            ("https://example.com/example.zip", "", ""),
            ("", SHA256, ""),
            ("https://example.com/example.zip", SHA256, "10.5281/zenodo.123"),
        ],
    )
    def test_rejects_incomplete_or_ambiguous_download_source(self, url: str, known_hash: str, doi: str) -> None:
        dataset = manifest.TestDataset(
            key="example",
            group="group",
            extracted_dir="example",
            source="example",
            url=url,
            known_hash=known_hash,
            doi=doi,
        )

        with pytest.raises(ValueError, match="either doi|both an archive"):
            manifest.validate_datasets((dataset,))

    def test_rejects_manifest_without_datasets_array(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text("datasets = { key = 'example' }\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"\[\[datasets\]\] array"):
            manifest.load_datasets(manifest_path)

    def test_rejects_invalid_toml(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text("[[datasets]\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid dataset manifest TOML"):
            manifest.load_datasets(manifest_path)

    def test_rejects_unknown_root_manifest_field(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text("version = 1\ndatasets = []\n", encoding="utf-8")

        with pytest.raises(ValueError, match="unknown root field"):
            manifest.load_datasets(manifest_path)

    def test_rejects_missing_required_manifest_field(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text(
            f"""
[[datasets]]
key = "example"
group = "group"
extracted_dir = "example"
url = "https://example.com/example.zip"
known_hash = "{SHA256}"
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="missing required field"):
            manifest.load_datasets(manifest_path)

    def test_rejects_unknown_manifest_field(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "datasets.toml"
        manifest_path.write_text(
            f"""
[[datasets]]
key = "example"
group = "group"
extracted_dir = "example"
source = "example"
url = "https://example.com/example.zip"
known_hash = "{SHA256}"
checksum = "unexpected"
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="unknown field"):
            manifest.load_datasets(manifest_path)


class TestDownloadDataset:
    def test_skips_existing_dataset_without_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dataset = _make_dataset("existing")
        (tmp_path / dataset.extracted_dir).mkdir()
        monkeypatch.setattr(
            download_test_data,
            "_fetch_dataset",
            lambda *args: pytest.fail("download should have been skipped"),
        )

        download_test_data.download_dataset(dataset, tmp_path, force=False)

        assert "Skipping existing" in capsys.readouterr().out

    def test_replaces_existing_dataset_with_force(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        dataset = _make_dataset("force")
        target = tmp_path / dataset.extracted_dir
        target.mkdir()
        (target / "old.txt").write_text("old", encoding="utf-8")

        def fetch(dataset: object, temporary_path: Path, extracted_path: Path) -> None:
            (extracted_path / "payload.txt").write_text("ok", encoding="utf-8")

        monkeypatch.setattr(download_test_data, "_fetch_dataset", fetch)

        download_test_data.download_dataset(dataset, tmp_path, force=True)

        assert not (target / "old.txt").exists()
        assert (target / "payload.txt").read_text(encoding="utf-8") == "ok"

    def test_wraps_pooch_download_errors(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        dataset = _make_dataset("error")

        def fetch(dataset: object, temporary_path: Path, extracted_path: Path) -> None:
            raise requests.HTTPError("403 Client Error")

        monkeypatch.setattr(download_test_data, "_fetch_dataset", fetch)

        with pytest.raises(download_test_data.DatasetDownloadError, match="error: 403 Client Error"):
            download_test_data.download_dataset(dataset, tmp_path, force=False)

    def test_reports_download_without_expected_contents(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        dataset = _make_dataset("empty")
        monkeypatch.setattr(download_test_data, "_fetch_dataset", lambda *args: None)

        with pytest.raises(download_test_data.DatasetDownloadError, match="expected directory"):
            download_test_data.download_dataset(dataset, tmp_path, force=False)

        assert not (tmp_path / dataset.extracted_dir).exists()


class TestMain:
    def test_downloads_multiple_selected_datasets(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        first = _make_dataset("first")
        second = _make_dataset("second")
        third = _make_dataset("third")
        attempted: list[str] = []

        def download_dataset(dataset: Any, output: Path, force: bool) -> None:
            attempted.append(dataset.key)
            assert output == tmp_path
            assert not force

        monkeypatch.setattr(download_test_data, "DATASETS", (first, second, third))
        monkeypatch.setattr(download_test_data, "download_dataset", download_dataset)

        download_test_data.main(["--output", str(tmp_path), "--dataset", "first", "--dataset", "third"])

        assert attempted == ["first", "third"]

    def test_lists_available_datasets(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dataset = _make_dataset("listed", group="group", extracted_dir="listed-dir", source="listed source")
        monkeypatch.setattr(download_test_data, "DATASETS", (dataset,))

        download_test_data.main(["--list"])

        assert capsys.readouterr().out == "listed\tgroup\tlisted-dir\tlisted source\n"

    def test_continues_after_failure_then_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        first = _make_dataset("first")
        second = _make_dataset("second")
        attempted: list[str] = []

        def download_dataset(dataset: Any, output: Path, force: bool) -> None:
            attempted.append(dataset.key)
            if dataset == first:
                raise download_test_data.DatasetDownloadError(dataset, "HTTP 403 Forbidden")

        monkeypatch.setattr(download_test_data, "DATASETS", (first, second))
        monkeypatch.setattr(download_test_data, "download_dataset", download_dataset)

        with pytest.raises(SystemExit) as exc_info:
            download_test_data.main(["--output", str(tmp_path)])

        assert exc_info.value.code == 1
        assert attempted == ["first", "second"]
        output = capsys.readouterr().out
        assert "Failed to download 1 dataset(s)" in output
        assert "first" in output
