"""Validate DBiT public options and final reader-owned table linkage."""

from __future__ import annotations

import warnings
from functools import partial
from typing import TYPE_CHECKING

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from spatialdata.models import get_table_keys

import spatialdata_io.readers.dbit._reader as dbit_reader
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table
from spatialdata_io.readers.dbit import dbit
from spatialdata_io.readers.dbit._reader import _validate_options
from tests.integration.readers.dbit.test_reader import _write_layout

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from pytest_mock import MockerFixture
    from spatialdata import SpatialData

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers._utils.table import SparseDuplicateSummary


class TestValidateOptions:
    """Make names, positive square widths, and boolean border choices explicit."""

    @pytest.mark.parametrize(
        "dataset_id",
        ["", "has space", "a/b", "a\\b", "a:b", "__private", ".", "..", "table", "TABLE", "TaBlE", "sample\x00"],
    )
    def test_invalid_dataset_id(self, dataset_id: str) -> None:
        with pytest.raises(ValueError, match="dataset_id must be a valid SpatialData element name"):
            _validate_options(dataset_id, include_border=True, border_scale=1.0)

    @pytest.mark.parametrize("dataset_id", [7, b"sample", ["sample"]])
    def test_dataset_id_type(self, dataset_id: object) -> None:
        with pytest.raises(TypeError, match="dataset_id must be a string; found "):
            _validate_options(dataset_id, include_border=True, border_scale=1.0)  # pyrefly: ignore[bad-argument-type] - Exercise untrusted caller types.

    @pytest.mark.parametrize(
        "dataset_id", [None, "sample", "sample_1-2.3", "_x", "...", "tables", "table_1", "échantillon"]
    )
    def test_valid_dataset_id(self, dataset_id: str | None) -> None:
        _validate_options(dataset_id, include_border=True, border_scale=1.0)

    @pytest.mark.parametrize("dataset_id", ["sample_1-2.3", "_x", "...", "tables", "échantillon"])
    def test_valid_dataset_id_is_accepted_by_spatialdata(self, tmp_path: Path, dataset_id: str) -> None:
        _write_layout(tmp_path)

        result = dbit(tmp_path, dataset_id=dataset_id)
        result.write(tmp_path / "out.zarr")

        assert list(result.shapes) == [dataset_id]

    @pytest.mark.parametrize("include_border", [0, 1, "true", None])
    def test_invalid_boolean(self, include_border: int | str | None) -> None:
        with pytest.raises(TypeError, match="include_border"):
            _validate_options("sample", include_border=include_border, border_scale=1.0)  # pyrefly: ignore[bad-argument-type] - Exercise untrusted caller types.

    @pytest.mark.parametrize("border_scale", [True, "1", None])
    def test_invalid_scale_type(self, *, border_scale: bool | str | None) -> None:
        with pytest.raises(TypeError, match="border_scale"):
            _validate_options("sample", include_border=True, border_scale=border_scale)  # pyrefly: ignore[bad-argument-type] - Exercise untrusted caller types.

    @pytest.mark.parametrize("include_border", [True, False])
    @pytest.mark.parametrize("border_scale", [-1.0, 4.0, float("nan"), float("inf"), float("-inf")])
    def test_invalid_scale_range(self, *, include_border: bool, border_scale: float) -> None:
        with pytest.raises(ValueError, match="border_scale"):
            _validate_options("sample", include_border=include_border, border_scale=border_scale)

    @pytest.mark.parametrize("include_border", [True, False])
    @pytest.mark.parametrize("border_scale", [0, 1.0, 3.99])
    def test_valid_scale(self, *, include_border: bool, border_scale: float) -> None:
        _validate_options("sample", include_border=include_border, border_scale=border_scale)


class TestDbit:
    """Exercise validation precedence, inferred names, and final exact membership."""

    @pytest.mark.parametrize(
        ("options", "error", "match"),
        [
            ({"dataset_id": 7, "include_border": 1}, TypeError, "dataset_id"),
            ({"include_border": 1, "border_scale": "1"}, TypeError, "include_border"),
            ({"border_scale": -1.0, "invalid_geometries": "repair"}, ValueError, "border_scale"),
            ({"invalid_geometries": "repair", "data_axes": 1}, ValueError, "invalid_geometries"),
            ({"data_axes": 1, "tiff_series": -1}, TypeError, "data_axes"),
            ({"tiff_series": -1, "tiff_level": "0"}, ValueError, "tiff_series"),
            ({"tiff_level": "0", "scale_factors": (1,)}, TypeError, "tiff_level"),
            ({"chunks": 0, "path_policy": "lenient"}, ValueError, "chunks"),
            ({"path_policy": "lenient"}, ValueError, "path_policy"),
        ],
    )
    def test_options_are_validated_in_signature_order_before_io(
        self, tmp_path: Path, options: dict[str, object], error: type[Exception], match: str
    ) -> None:
        # Any filesystem access would fail first with FileNotFoundError.
        with pytest.raises(error, match=match):
            dbit(tmp_path / "absent", **options)  # pyrefly: ignore[bad-argument-type] - Untrusted caller input.

    def test_invalid_geometry_policy_precedes_io(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid_geometries"):
            dbit(tmp_path / "absent", invalid_geometries="repair")  # pyrefly: ignore[bad-argument-type] - Invalid runtime input.

    def test_invalid_grid_option_fails_before_discovery(self, tmp_path: Path, mocker: MockerFixture) -> None:
        discovery = mocker.spy(dbit_reader, "discover_layout")

        with pytest.raises(ValueError, match="border_scale"):
            dbit(tmp_path / "absent", include_border=False, border_scale=float("nan"))

        discovery.assert_not_called()

    def test_invalid_image_option_fails_before_discovery(self, tmp_path: Path, mocker: MockerFixture) -> None:
        discovery = mocker.spy(dbit_reader, "discover_layout")

        with pytest.raises(ValueError, match="data_axes"):
            dbit(tmp_path / "absent", data_axes="YYX")

        discovery.assert_not_called()

    def test_invalid_raster_option_fails_before_discovery(self, tmp_path: Path, mocker: MockerFixture) -> None:
        discovery = mocker.spy(dbit_reader, "discover_layout")

        with pytest.raises(ValueError, match="chunks"):
            dbit(tmp_path / "absent", chunks=0)

        discovery.assert_not_called()

    @pytest.mark.parametrize(
        "read",
        [
            partial(dbit, data_axes="YX"),
            partial(dbit, tiff_series=0),
            partial(dbit, tiff_level=1),
            partial(dbit, scale_factors=(2,)),
            partial(dbit, chunks=8),
            partial(dbit, chunks=(1, 512, 512)),
        ],
        ids=["data_axes", "tiff_series", "tiff_level", "scale_factors", "chunks", "chunk_tuple"],
    )
    def test_nondefault_image_options_require_image(self, tmp_path: Path, read: Callable[[Path], SpatialData]) -> None:
        _write_layout(tmp_path)

        with pytest.raises(ValueError, match="require an image"):
            read(tmp_path)

    def test_explicit_default_image_options_need_no_image(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)

        result = dbit(tmp_path, tiff_series=None, tiff_level=0, scale_factors=(), chunks=(1, 1024, 1024))

        assert dict(result.images) == {}

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("invalid name.h5ad", "invalid_name"),
            ("..h5ad", "unnamed"),
            ("...h5ad", "unnamed"),
            ("__x.h5ad", "_x"),
            ("table.h5ad", "table_"),
            ("TABLE.h5ad", "TABLE_"),
        ],
    )
    def test_unusable_filename_stem_is_sanitized_with_one_warning(
        self, tmp_path: Path, name: str, expected: str
    ) -> None:
        _write_layout(tmp_path)
        (tmp_path / "sample.h5ad").rename(tmp_path / name)
        stem = name.removesuffix(".h5ad").removesuffix(".H5AD")

        with pytest.warns(UserWarning, match="filename stem") as caught:
            result = dbit(tmp_path)

        message = (
            f"dbit: the H5AD filename stem {stem!r} is not a usable SpatialData element name; using {expected!r}. "
            "Pass dataset_id to choose the name explicitly."
        )
        assert [str(warning.message) for warning in caught] == [message]
        assert caught[0].filename == __file__
        assert list(result.shapes) == [expected]
        assert get_table_keys(result.tables["table"]) == (expected, "region", "pixel_id")
        assert result.tables["table"].obs["sample"].unique().tolist() == [expected]

    def test_explicit_name_overrides_invalid_filename_stem_without_warning(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        (tmp_path / "sample.h5ad").rename(tmp_path / "invalid name.h5ad")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = dbit(tmp_path, dataset_id="valid")

        assert list(result.shapes) == ["valid"]
        assert get_table_keys(result.tables["table"]) == ("valid", "region", "pixel_id")

    @pytest.mark.parametrize(
        "linkage",
        [
            None,
            False,
            "",
            {},
            {"region": 42},
            {"region": "old", "region_key": "region", "instance_key": "pixel_id"},
            {"region": "old", "region_key": "custom_region", "instance_key": "custom_id"},
        ],
    )
    def test_final_linkage_replaces_inherited_metadata_once(
        self, tmp_path: Path, mocker: MockerFixture, *, linkage: bool | str | dict[str, str | int] | None
    ) -> None:
        """Link exactly once with final independent targets and no inherited declaration."""
        _write_layout(tmp_path)
        source = ad.read_h5ad(tmp_path / "sample.h5ad")
        source.uns["spatialdata_attrs"] = linkage
        source.obs["region"] = pd.Categorical(["old", "old"])
        source.obs["pixel_id"] = [90, 91]
        mocker.patch("spatialdata_io.readers.dbit._tables.ad.read_h5ad", return_value=source)

        def checked_linkage(  # noqa: PLR0913 - mirrors the keyword-only linkage signature
            table: ad.AnnData,
            target: TableLinkage,
            /,
            *,
            context: ReaderErrorContext,
            path: Path,
            expression_layers: tuple[str, ...],
            expected_instance_ids: Mapping[str, pd.Index],
            duplicate_sums: list[SparseDuplicateSummary],
        ) -> ad.AnnData:
            """Inspect the pre-link table before the real parser attaches new metadata."""
            assert "spatialdata_attrs" not in table.uns
            assert table.obs["region"].tolist() == ["sample", "sample"]
            # Linkage sees source order; pixel_id is each row's sorted position.
            np.testing.assert_array_equal(table.obs["pixel_id"], [1, 0])
            return parse_linked_table(
                table,
                target,
                context=context,
                path=path,
                expression_layers=expression_layers,
                expected_instance_ids=expected_instance_ids,
                duplicate_sums=duplicate_sums,
            )

        parse = mocker.patch.object(dbit_reader, "parse_linked_table", side_effect=checked_linkage)
        discovery = mocker.spy(dbit_reader, "discover_layout")
        grid = mocker.spy(dbit_reader, "make_grid")

        result = dbit(tmp_path)

        discovery.assert_called_once()
        grid.assert_called_once()
        parse.assert_called_once()
        assert parse.call_args.kwargs["expression_layers"] == ("counts",)
        expected_ids = parse.call_args.kwargs["expected_instance_ids"]
        assert list(expected_ids) == ["sample"]
        pd.testing.assert_index_equal(expected_ids["sample"], result.shapes["sample"].index)
        assert expected_ids["sample"] is grid.spy_return.index
        assert get_table_keys(result.tables["table"]) == ("sample", "region", "pixel_id")
        np.testing.assert_array_equal(result.tables["table"].obs["pixel_id"], result.shapes["sample"].index)

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

    def test_linkage_failure_carries_table_context(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        table = ad.read_h5ad(tmp_path / "sample.h5ad")
        table.X = np.full(table.shape, np.nan)
        table.write_h5ad(tmp_path / "sample.h5ad")

        with pytest.raises(ValueError, match="finite") as caught:
            dbit(tmp_path)

        notes = caught.value.__notes__
        assert any("dbit" in note and str((tmp_path / "sample.h5ad").resolve()) in note for note in notes)

    @pytest.mark.parametrize("slot", ["X", "counts"])
    @pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
    def test_nonfinite_expression_fails_with_table_context(self, tmp_path: Path, slot: str, value: float) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "sample.h5ad"
        source = ad.read_h5ad(path)
        expression = sparse.csr_array(np.array([[value, 0], [0, 1]], dtype=np.float64))
        if slot == "X":
            source.X = expression
        else:
            source.layers[slot] = expression
        source.write_h5ad(path)

        with pytest.raises(ValueError, match="finite") as caught:
            dbit(tmp_path)

        assert any("expression table" in note and str(path.resolve()) in note for note in caught.value.__notes__)

    @pytest.mark.parametrize("slot", ["X", "counts"])
    def test_malformed_sparse_storage_fails_before_row_selection(
        self, tmp_path: Path, mocker: MockerFixture, slot: str
    ) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "sample.h5ad"
        with h5py.File(path, "r+") as source:
            group = "X" if slot == "X" else "layers/counts"
            source[f"{group}/indices"][0] = 2
        copy = mocker.spy(ad.AnnData, "copy")

        with pytest.raises(ValueError, match="active indices") as caught:
            dbit(tmp_path)

        copy.assert_not_called()
        assert any("expression table" in note and str(path.resolve()) in note for note in caught.value.__notes__)

    def test_dense_expression_is_sparse_before_rows_are_sorted(self, tmp_path: Path, mocker: MockerFixture) -> None:
        _write_layout(tmp_path, sparse_source=False)
        original_copy = ad.AnnData.copy
        copied: list[bool] = []

        def checked_copy(table: ad.AnnData, filename: str | Path | None = None) -> ad.AnnData:
            """Fail if sorting would materialize a second dense expression matrix."""
            assert isinstance(table.X, sparse.csr_array)
            assert isinstance(table.layers["counts"], sparse.csr_array)
            copied.append(True)
            return original_copy(table, filename)

        mocker.patch.object(ad.AnnData, "copy", checked_copy)

        result = dbit(tmp_path)

        assert copied == [True]
        expression = result.tables["table"].X
        assert isinstance(expression, sparse.csr_array)
        np.testing.assert_array_equal(expression.toarray(), [[1, 2], [3, 4]])

    def test_noncanonical_expression_sums_once_with_one_warning(self, tmp_path: Path) -> None:
        _write_layout(tmp_path)
        path = tmp_path / "sample.h5ad"
        source = ad.read_h5ad(path)
        source.X = sparse.csr_array(
            (np.array([1, 2, 4], dtype=np.int64), np.array([1, 1, 0]), np.array([0, 2, 3])), shape=(2, 2)
        )
        source.write_h5ad(path)

        with pytest.warns(UserWarning, match=r"^dbit sample\.h5ad: combined 1 duplicate sparse entry in X$") as caught:
            result = dbit(tmp_path)

        assert len(caught) == 1
        expression = result.tables["table"].X
        assert isinstance(expression, sparse.csr_array)
        assert expression.has_canonical_format
        assert expression.dtype == np.int64
        np.testing.assert_array_equal(expression.toarray(), [[4, 0], [0, 3]])
