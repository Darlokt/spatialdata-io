"""Validate DBiT public options and final reader-owned table linkage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from spatialdata.models import get_table_keys

import spatialdata_io.readers.dbit._reader as dbit_reader
from spatialdata_io.readers.dbit import dbit
from spatialdata_io.readers.dbit._reader import _validate_grid_options
from tests.integration.readers.dbit.test_reader import _write_layout

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestValidateGridOptions:
    """Make names, positive square widths, and boolean border choices explicit."""

    @pytest.mark.parametrize(
        "dataset_id", ["", "has space", "a/b", "a\\b", "__private", ".", "..", "table", "TABLE", "TaBlE", 7]
    )
    def test_invalid_dataset_id(self, dataset_id: str | int) -> None:
        with pytest.raises(ValueError, match="dataset_id"):
            _validate_grid_options(dataset_id, include_border=True, border_scale=1.0)  # type: ignore[bad-argument-type, arg-type]  # Exercise untrusted caller types.

    @pytest.mark.parametrize("dataset_id", [None, "sample", "sample_1-2.3", "échantillon"])
    def test_valid_dataset_id(self, dataset_id: str | None) -> None:
        _validate_grid_options(dataset_id, include_border=True, border_scale=1.0)

    @pytest.mark.parametrize("include_border", [0, 1, "true", None])
    def test_invalid_boolean(self, include_border: int | str | None) -> None:
        with pytest.raises(TypeError, match="include_border"):
            _validate_grid_options("sample", include_border=include_border, border_scale=1.0)  # type: ignore[bad-argument-type, arg-type]  # Exercise untrusted caller types.

    @pytest.mark.parametrize("border_scale", [True, "1", None])
    def test_invalid_scale_type(self, *, border_scale: bool | str | None) -> None:
        with pytest.raises(TypeError, match="border_scale"):
            _validate_grid_options("sample", include_border=True, border_scale=border_scale)  # type: ignore[bad-argument-type, arg-type]  # Exercise untrusted caller types.

    @pytest.mark.parametrize("include_border", [True, False])
    @pytest.mark.parametrize("border_scale", [-1.0, 4.0, float("nan"), float("inf"), float("-inf")])
    def test_invalid_scale_range(self, *, include_border: bool, border_scale: float) -> None:
        with pytest.raises(ValueError, match="border_scale"):
            _validate_grid_options("sample", include_border=include_border, border_scale=border_scale)

    @pytest.mark.parametrize("include_border", [True, False])
    @pytest.mark.parametrize("border_scale", [0, 1.0, 3.99])
    def test_valid_scale(self, *, include_border: bool, border_scale: float) -> None:
        _validate_grid_options("sample", include_border=include_border, border_scale=border_scale)


class TestDbit:
    """Exercise validation precedence, inferred names, and final exact membership."""

    def test_invalid_grid_option_fails_before_discovery(self, tmp_path: Path, mocker: MockerFixture) -> None:
        discovery = mocker.spy(dbit_reader, "_discover_layout")

        with pytest.raises(ValueError, match="border_scale"):
            dbit(tmp_path / "absent", include_border=False, border_scale=float("nan"))

        discovery.assert_not_called()

    def test_invalid_image_option_fails_before_discovery(self, tmp_path: Path, mocker: MockerFixture) -> None:
        discovery = mocker.spy(dbit_reader, "_discover_layout")

        with pytest.raises(ValueError, match="data_axes"):
            dbit(tmp_path / "absent", data_axes="YYX")

        discovery.assert_not_called()

    @pytest.mark.parametrize(
        ("data_axes", "tiff_series", "tiff_level", "chunks"),
        [("YX", 0, 0, None), (None, 1, 0, None), (None, 0, 1, None), (None, 0, 0, 8)],
    )
    def test_image_options_require_image(
        self, tmp_path: Path, data_axes: str | None, tiff_series: int, tiff_level: int, chunks: int | None
    ) -> None:
        _write_layout(tmp_path)

        with pytest.raises(ValueError, match="require an image"):
            dbit(tmp_path, data_axes=data_axes, tiff_series=tiff_series, tiff_level=tiff_level, chunks=chunks)

    @pytest.mark.parametrize("name", ["invalid name.h5ad", "..h5ad", "...h5ad", "table.h5ad", "TABLE.h5ad"])
    def test_inferred_name_is_validated_before_loading(self, tmp_path: Path, mocker: MockerFixture, name: str) -> None:
        _write_layout(tmp_path)
        (tmp_path / "sample.h5ad").rename(tmp_path / name)
        barcode_read = mocker.spy(dbit_reader, "_read_barcodes")
        table_read = mocker.spy(dbit_reader, "_read_table")

        with pytest.raises(ValueError, match="dataset_id"):
            dbit(tmp_path)

        barcode_read.assert_not_called()
        table_read.assert_not_called()

    def test_explicit_name_overrides_invalid_filename_stem(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        (tmp_path / "sample.h5ad").rename(tmp_path / "invalid name.h5ad")

        result = dbit(tmp_path, dataset_id="valid")

        assert list(result.shapes) == ["valid"]
        assert get_table_keys(result.tables["table"]) == ("valid", "region", "pixel_id")

    def test_stale_linkage_is_replaced_once_with_final_shape_membership(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        _write_layout(tmp_path)
        source = tmp_path / "sample.h5ad"
        original = ad.read_h5ad(source)
        original.obs["region"] = pd.Categorical(["old", "old"])
        original.obs["pixel_id"] = [90, 91]
        original.uns["spatialdata_attrs"] = {"region": ["old"], "region_key": "region", "instance_key": "pixel_id"}
        original.write_h5ad(source)
        parse = mocker.spy(dbit_reader, "parse_linked_table")
        discovery = mocker.spy(dbit_reader, "_discover_layout")

        result = dbit(tmp_path)

        discovery.assert_called_once()
        parse.assert_called_once()
        assert parse.call_args.kwargs["expression_layers"] == ("counts",)
        expected_ids = parse.call_args.kwargs["expected_instance_ids"]
        assert list(expected_ids) == ["sample"]
        pd.testing.assert_index_equal(expected_ids["sample"], result.shapes["sample"].index)
        assert expected_ids["sample"] is not result.shapes["sample"].index

        table = result.tables["table"]
        assert get_table_keys(table) == ("sample", "region", "pixel_id")
        assert table.obs["region"].tolist() == ["sample", "sample"]
        np.testing.assert_array_equal(table.obs["pixel_id"], result.shapes["sample"].index)

        retained = ad.read_h5ad(source)
        assert retained.obs["region"].tolist() == ["old", "old"]
        np.testing.assert_array_equal(retained.obs["pixel_id"], [90, 91])
        assert retained.uns["spatialdata_attrs"]["region"] == ["old"]
        assert "array_A" not in retained.obs

    def test_table_without_counts_layer_has_no_expression_layer_override(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        _write_layout(tmp_path)
        source = tmp_path / "sample.h5ad"
        original = ad.read_h5ad(source)
        del original.layers["counts"]
        original.write_h5ad(source)
        parse = mocker.spy(dbit_reader, "parse_linked_table")

        result = dbit(tmp_path)

        assert parse.call_args.kwargs["expression_layers"] == ()
        assert "counts" not in result.tables["table"].layers

        quality = result.tables["table"].layers["quality"]
        assert isinstance(quality, np.ndarray)
        np.testing.assert_allclose(quality, [[0.1, 0.2], [0.3, 0.4]])

    def test_linkage_failure_preserves_artifact_context(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _write_layout(tmp_path)
        mocker.patch.object(dbit_reader, "parse_linked_table", side_effect=ValueError("injected linkage failure"))

        with pytest.raises(ValueError, match="injected linkage failure") as caught:
            dbit(tmp_path)

        notes = caught.value.__notes__
        assert any("dbit" in note and str(tmp_path / "sample.h5ad") in note for note in notes)
