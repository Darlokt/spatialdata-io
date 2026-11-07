"""Tests for the public Visium reader boundary."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import pytest

import spatialdata_io
from spatialdata_io._raster_defaults import DEFAULT_CHUNKS, DEFAULT_SCALE_FACTORS
from spatialdata_io.readers._utils.paths import ArtifactDirectory, ArtifactFile
from spatialdata_io.readers.visium import visium
from spatialdata_io.readers.visium._layout import H5Counts, MtxCounts
from spatialdata_io.readers.visium._metadata import CountMetadata
from spatialdata_io.readers.visium._reader import _derived_dataset_id

if TYPE_CHECKING:
    from pathlib import Path

# One invalid value per option, in signature order, with the error it raises.
_INVALID_OPTIONS: list[tuple[str, object, type[Exception], str]] = [
    ("dataset_id", "a b", ValueError, "dataset_id must be a valid SpatialData element name"),
    ("fullres_image_file", "full.bmp", ValueError, "fullres_image_file must be a TIFF/BigTIFF"),
    ("make_var_names_unique", 1, TypeError, "make_var_names_unique must be a boolean"),
    ("data_axes", ("Y", "X"), TypeError, "data_axes must be a string"),
    ("tiff_series", -1, ValueError, "tiff_series must be nonnegative"),
    ("scale_factors", (1,), ValueError, "scale_factors must contain factors of at least 2"),
    ("chunks", {"c": 1, "y": 2, "x": 2}, TypeError, "chunks must be a positive integer"),
    ("path_policy", "loose", ValueError, "path_policy"),
]


class TestVisium:
    """Keep the deliberately small public API and early option validation exact."""

    def test_public_signature_contains_only_format_concerns(self) -> None:
        signature = inspect.signature(visium)

        assert list(signature.parameters) == [
            "path",
            "dataset_id",
            "counts_source",
            "fullres_image_file",
            "positions_file",
            "scale_factors_file",
            "make_var_names_unique",
            "data_axes",
            "tiff_series",
            "scale_factors",
            "chunks",
            "path_policy",
        ]
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for name, parameter in signature.parameters.items()
            if name != "path"
        )
        assert signature.parameters["make_var_names_unique"].default is True
        assert signature.parameters["data_axes"].default is None
        assert signature.parameters["data_axes"].annotation == "str | None"
        assert signature.parameters["tiff_series"].default is None
        assert signature.parameters["scale_factors"].default == DEFAULT_SCALE_FACTORS
        assert signature.parameters["chunks"].default == DEFAULT_CHUNKS
        assert signature.parameters["chunks"].annotation == "int | tuple[int, int, int]"
        assert signature.parameters["path_policy"].default == "strict"

    def test_lazy_root_export_is_the_same_reader(self) -> None:
        assert spatialdata_io.visium is visium

    @pytest.mark.parametrize(("option", "value", "error", "match"), _INVALID_OPTIONS)
    def test_rejects_each_invalid_option_before_filesystem_work(
        self, option: str, value: object, error: type[Exception], match: str
    ) -> None:
        options: dict[str, Any] = {option: value}
        if option in {"data_axes", "tiff_series", "scale_factors"}:
            options["fullres_image_file"] = "full.tif"

        with pytest.raises(error, match=match):
            visium("missing", **options)

    @pytest.mark.parametrize("index", range(len(_INVALID_OPTIONS) - 1))
    def test_options_are_validated_in_signature_order(self, index: int) -> None:
        first, second = _INVALID_OPTIONS[index], _INVALID_OPTIONS[index + 1]
        options: dict[str, Any] = {first[0]: first[1], second[0]: second[1]}

        with pytest.raises(first[2], match=first[3]):
            visium("missing", **options)

    @pytest.mark.parametrize(
        ("dataset_id", "error", "match"),
        [
            (1, TypeError, "dataset_id must be a string; found int"),
            ("", ValueError, "valid SpatialData element name"),
            ("a/b", ValueError, "valid SpatialData element name"),
            ("__x", ValueError, "valid SpatialData element name"),
            ("TaBlE", ValueError, r"other than \['table'\]"),
        ],
    )
    def test_rejects_invalid_dataset_ids(self, dataset_id: object, error: type[Exception], match: str) -> None:
        with pytest.raises(error, match=match):
            visium("missing", dataset_id=dataset_id)  # pyrefly: ignore[bad-argument-type] - Invalid runtime inputs.

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"data_axes": "CYX"}, "require a TIFF-family fullres_image_file"),
            ({"tiff_series": 0, "fullres_image_file": "full.png"}, "require a TIFF-family fullres_image_file"),
            ({"scale_factors": (2,)}, "scale_factors require fullres_image_file"),
        ],
    )
    def test_rejects_image_options_without_a_suitable_image_before_filesystem_work(
        self, options: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            visium("missing", **options)


class TestDerivedDatasetId:
    """Name the dataset from the count source, repairing unusable names."""

    def test_prefers_the_library_id_over_the_filename_prefix(self, tmp_path: Path) -> None:
        counts = H5Counts(ArtifactFile(tmp_path / "x.h5"), "prefix", CountMetadata("library", None, None))

        assert _derived_dataset_id(counts) == "library"

    @pytest.mark.parametrize("metadata", [CountMetadata(None, None, None), None])
    def test_falls_back_to_the_filename_prefix(self, tmp_path: Path, metadata: CountMetadata | None) -> None:
        counts = (
            MtxCounts(ArtifactDirectory(tmp_path), None, compressed=True, filename_prefix="prefix")
            if metadata is None
            else H5Counts(ArtifactFile(tmp_path / "x.h5"), "prefix", metadata)
        )

        assert _derived_dataset_id(counts) == "prefix"

    @pytest.mark.parametrize(
        ("library_id", "prefix", "expected", "source"),
        [
            ("my library", None, "my_library", "visium: the HDF5 library ID 'my library'"),
            ("Table", None, "Table_", "visium: the HDF5 library ID 'Table'"),
            (None, "V1 Mouse", "V1_Mouse", "visium: the count filename prefix 'V1 Mouse'"),
        ],
    )
    def test_sanitizes_unusable_names_with_one_warning(
        self, tmp_path: Path, library_id: str | None, prefix: str | None, expected: str, source: str
    ) -> None:
        counts = H5Counts(ArtifactFile(tmp_path / "x.h5"), prefix, CountMetadata(library_id, None, None))

        with pytest.warns(UserWarning, match=rf"^{source} is not a usable .* Pass dataset_id") as caught:
            assert _derived_dataset_id(counts) == expected

        assert len(caught) == 1

    def test_requires_dataset_id_without_a_source_name(self, tmp_path: Path) -> None:
        counts = MtxCounts(ArtifactDirectory(tmp_path), None, compressed=True, filename_prefix=None)

        with pytest.raises(ValueError, match="dataset_id is required"):
            _derived_dataset_id(counts)
