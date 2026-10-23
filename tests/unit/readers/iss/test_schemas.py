"""ISS annotation extensions, exact IDs, and reserved-field collision contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.iss._schemas import (
    _validate_index_ids,
    _validate_instance_ids,
    _validate_output_column,
    _validate_table,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestValidateTable:
    """Reject ambiguous names without constraining named-field order or extras."""

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
