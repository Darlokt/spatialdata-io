"""ISS annotation extensions, exact IDs, and reserved-field collision contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.iss._schemas import (
    _validate_expression,
    _validate_index_ids,
    _validate_instance_ids,
    _validate_output_column,
    _validate_table,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestValidateExpression:
    """Guard declared expression slots without mutating compressed source storage."""

    def test_dense_expression_with_absent_counts_is_preserved(self, tmp_path: Path) -> None:
        expression = np.asarray([[100, 100], [0, 0]], dtype=np.int8)
        table = ad.AnnData(expression)
        _validate_expression(table, path=ArtifactFile(tmp_path / "source.h5ad"))
        assert table.X is expression
        assert "counts" not in table.layers
        np.testing.assert_array_equal(expression, [[100, 100], [0, 0]])

    @pytest.mark.parametrize("dtype", ["float32", "bool"])
    def test_noninteger_duplicates_keep_their_existing_policy(self, tmp_path: Path, dtype: str) -> None:
        expression = sparse.csr_array((np.asarray([1, 1], dtype=dtype), [0, 0], [0, 2, 2]), shape=(2, 2))
        table = ad.AnnData(expression)
        table.layers["counts"] = expression.copy()
        _validate_expression(table, path=ArtifactFile(tmp_path / "source.h5ad"))
        assert table.X is expression
        for actual in (expression, table.layers["counts"]):
            assert isinstance(actual, sparse.csr_array)
            assert actual.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(actual.data, [1, 1])
            np.testing.assert_array_equal(actual.indices, [0, 0])

    @pytest.mark.parametrize("slot", ["X", "counts"])
    @pytest.mark.parametrize("storage", ["csr", "csc"])
    @pytest.mark.parametrize(
        "case",
        [
            ("int8", [100, 100], True),
            ("uint8", [200, 200], True),
            ("int64", [2**63 - 1, 1], True),
            ("uint64", [2**63, 2**63], True),
            ("int64", [2**63 - 2, 1], False),
            ("uint64", [2**64 - 2, 1], False),
            ("int8", [120, 120, -120], False),
        ],
    )
    def test_integer_duplicate_sums_before_normalization(
        self, tmp_path: Path, slot: str, storage: str, case: tuple[str, list[int], bool]
    ) -> None:
        dtype, values, _ = case
        constructor = sparse.csr_array if storage == "csr" else sparse.csc_array
        expression = constructor(
            (np.asarray(values, dtype=dtype), np.zeros(len(values), dtype=np.int32), [0, len(values), len(values)]),
            shape=(2, 2),
        )
        table = ad.AnnData(np.ones((2, 2)))
        if slot == "X":
            table.X = expression
        else:
            table.layers[slot] = expression
        original = (expression.data.copy(), expression.indices.copy(), expression.indptr.copy())
        path = ArtifactFile(tmp_path / "source.h5ad")

        _validate_expression(table, path=path)

        assert (table.X if slot == "X" else table.layers[slot]) is expression
        for actual, expected in zip((expression.data, expression.indices, expression.indptr), original, strict=True):
            np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("storage", ["csr", "csc"])
    def test_unsorted_unique_integer_entries_are_not_duplicate_sums(self, tmp_path: Path, storage: str) -> None:
        constructor = sparse.csr_array if storage == "csr" else sparse.csc_array
        expression = constructor((np.asarray([127, 127], dtype=np.int8), [1, 0], [0, 2, 2]), shape=(2, 2))
        table = ad.AnnData(expression)
        _validate_expression(table, path=ArtifactFile(tmp_path / "source.h5ad"))
        np.testing.assert_array_equal(expression.data, [127, 127])
        np.testing.assert_array_equal(expression.indices, [1, 0])
        assert not expression.has_sorted_indices

    def test_unrelated_integer_layer_is_outside_expression_contract(self, tmp_path: Path) -> None:
        table = ad.AnnData(np.ones((2, 2)))
        expression = sparse.csr_array((np.asarray([100, 100], dtype=np.int8), [0, 0], [0, 2, 2]), shape=(2, 2))
        table.layers["metadata"] = expression
        _validate_expression(table, path=ArtifactFile(tmp_path / "source.h5ad"))
        assert table.layers["metadata"] is expression
        np.testing.assert_array_equal(expression.data, [100, 100])

    @pytest.mark.parametrize("slot", ["X", "counts"])
    @pytest.mark.parametrize("storage", ["csr", "csc"])
    def test_legacy_compressed_matrix_overflow(self, tmp_path: Path, slot: str, storage: str) -> None:
        constructor = sparse.csr_matrix if storage == "csr" else sparse.csc_matrix
        expression = constructor((np.asarray([100, 100], dtype=np.int8), [0, 0], [0, 2, 2]), shape=(2, 2))
        table = ad.AnnData(np.ones((2, 2)))
        if slot == "X":
            table.X = expression
        else:
            table.layers[slot] = expression
        _validate_expression(table, path=ArtifactFile(tmp_path / "source.h5ad"))
        np.testing.assert_array_equal(expression.data, [100, 100])

    @pytest.mark.parametrize("invalid", ["range", "dtype"])
    def test_malformed_compressed_storage_preserves_error_and_context(
        self, tmp_path: Path, mocker: MockerFixture, invalid: str
    ) -> None:
        expression = sparse.csr_array(np.eye(2, dtype=np.int64))
        table = ad.AnnData(expression)
        if invalid == "range":
            expression.indices[0] = 2
        else:
            mocker.patch.object(expression, "indices", np.asarray([0, 1], dtype=np.float64))
        original = expression.indices.copy()
        path = ArtifactFile(tmp_path / "source.h5ad")
        with pytest.raises(ValueError if invalid == "range" else TypeError, match="indices") as caught:
            _validate_expression(table, path=path)
        notes = " ".join(caught.value.__notes__)
        assert "iss" in notes
        assert "X" in notes
        assert str(path) in notes
        np.testing.assert_array_equal(expression.indices, original)


class TestValidateTable:
    """Reject ambiguous names without constraining named-field order or extras."""

    def test_expression_duplicates_are_left_for_safe_normalization(self, tmp_path: Path) -> None:
        table = ad.AnnData(sparse.csr_array((np.asarray([100, 100], dtype=np.int8), [0, 0], [0, 2, 2]), shape=(2, 2)))
        _validate_table(table, path=ArtifactFile(tmp_path / "source.h5ad"))

    @pytest.mark.parametrize("destination", ["spatial_source", "spatialdata_io_positions"])
    def test_reserved_coordinate_destinations(self, tmp_path: Path, destination: str) -> None:
        table = ad.AnnData(np.ones((2, 2)))
        native = np.arange(6, dtype=np.int16).reshape(2, 3)
        table.obsm["spatial"] = native
        if destination == "spatialdata_io_positions":
            table.uns[destination] = {"coordinates": {"spatial": {"units": "micrometer"}}}
        else:
            table.obsm[destination] = native
        with pytest.raises(ReaderFormatError, match="occupied"):
            _validate_table(table, path=ArtifactFile(tmp_path / "source.h5ad"))
        assert table.obsm["spatial"] is native

    @pytest.mark.parametrize("axis", ["obs", "var"])
    def test_duplicate_columns(self, tmp_path: Path, axis: str) -> None:
        """Duplicate columns."""
        table = ad.AnnData(np.ones((2, 2)))
        frame = pd.DataFrame([[1, 2], [3, 4]], index=["0", "1"], columns=["same", "same"])
        setattr(table, axis, frame)
        with pytest.raises(ReaderFormatError, match="duplicate annotation columns"):
            _validate_table(table, path=ArtifactFile(tmp_path / "table.h5ad"))

    def test_extensions_and_reordered_columns_are_retained(self, tmp_path: Path) -> None:
        """Extensions and reordered columns are retained."""
        table = ad.AnnData(np.ones((2, 2)))
        table.obs["extra"] = pd.array([1, None], dtype="Int64")
        table.obs["label"] = pd.Categorical(["T", "B"])
        assert isinstance(table.obs, pd.DataFrame)
        before = table.obs.copy()
        _validate_table(table, path=ArtifactFile(tmp_path / "table.h5ad"))
        pd.testing.assert_frame_equal(table.obs, before)

    def test_missing_selected_instance_column(self, tmp_path: Path) -> None:
        """Missing selected instance column."""
        with pytest.raises(ReaderFormatError, match="missing observation column"):
            _validate_table(
                ad.AnnData(np.ones((1, 1))), path=ArtifactFile(tmp_path / "table.h5ad"), instance_key="missing"
            )


class TestValidateInstanceIds:
    """Check source integers independently of their physical nullable storage."""

    @pytest.mark.parametrize("dtype", ["uint64", "UInt64", "uint64[pyarrow]"])
    def test_uint64_limit(self, tmp_path: Path, dtype: str) -> None:
        """Uint64 limit."""
        _validate_instance_ids(
            pd.Series([2**64 - 1], dtype=dtype),
            label_dtype=np.dtype(np.uint64),
            path=ArtifactFile(tmp_path / "table.h5ad"),
        )

    @pytest.mark.parametrize("dtype", ["boolean", "float64", "Int64"])
    def test_nulls_rejected(self, tmp_path: Path, dtype: str) -> None:
        """Nulls rejected."""
        with pytest.raises(ReaderFormatError, match="null or noninteger"):
            _validate_instance_ids(
                pd.Series([None], dtype=dtype),
                label_dtype=np.dtype(np.uint64),
                path=ArtifactFile(tmp_path / "table.h5ad"),
            )

    @pytest.mark.parametrize("values", [[True, False], [1.0, 2.0], ["1", "2"], [0, 2], [-1, 2], [65536, 2]])
    def test_invalid_explicit_values(
        self, tmp_path: Path, values: list[bool] | list[int] | list[float] | list[str]
    ) -> None:
        """Invalid explicit values."""
        obs = pd.DataFrame({"cell_id": values})

        with pytest.raises(ReaderFormatError):
            _validate_instance_ids(
                obs["cell_id"],
                label_dtype=np.dtype(np.uint16),
                path=ArtifactFile(tmp_path / "counts.h5ad"),
            )


class TestValidateOutputColumn:
    """Compare reserved fields without float64 promotion of exact integers."""

    @pytest.mark.parametrize("existing", [2**53, float(2**53), None, "9007199254740993", True])
    def test_conflicts_with_exact_wide_id(self, tmp_path: Path, existing: float | str | None) -> None:
        """Conflicts with exact wide id."""
        obs = pd.DataFrame({"instance_id": [existing]})
        with pytest.raises(ReaderFormatError, match="conflicting reserved"):
            _validate_output_column(
                obs,
                "instance_id",
                pd.Series([2**53 + 1], dtype="uint64[pyarrow]"),
                path=ArtifactFile(tmp_path / "table.h5ad"),
            )

    @pytest.mark.parametrize("present", [True, False])
    def test_equal_or_absent_fields(self, tmp_path: Path, *, present: bool) -> None:
        """Equal or absent fields."""
        obs = pd.DataFrame({"instance_id": [2**64 - 1]}) if present else pd.DataFrame(index=[0])
        _validate_output_column(
            obs,
            "instance_id",
            pd.Series([2**64 - 1], dtype="uint64[pyarrow]"),
            path=ArtifactFile(tmp_path / "table.h5ad"),
        )


class TestValidateIndexIds:
    """Reject invalid parsed decimal IDs before constructing an Arrow integer array."""

    @pytest.mark.parametrize("value", [0, -1, 65536])
    def test_nonpositive_or_out_of_range(self, tmp_path: Path, value: int) -> None:
        """Respect label background and dtype bounds without float conversion."""
        with pytest.raises(ReaderFormatError, match="nonpositive or out-of-range"):
            _validate_index_ids([value], label_dtype=np.dtype(np.uint16), path=ArtifactFile(tmp_path / "table.h5ad"))
