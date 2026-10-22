"""ISS h5ad ownership, schema, and integer instance semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.iss._tables import _read_table, _set_instance_ids
from tests.integration.readers.iss.conftest import write_iss_dataset

if TYPE_CHECKING:
    from pathlib import Path


class TestReadTable:
    """Reject malformed scientific inputs before final linkage."""

    def test_read_failure_has_context(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.h5ad"
        path.write_bytes(b"not HDF5")
        with pytest.raises(OSError, match="file signature") as caught:
            _read_table(ArtifactFile(path), instance_key=None, label_dtype=np.dtype(np.uint16))
        assert any("iss" in note and str(path) in note for note in caught.value.__notes__)

    @pytest.mark.parametrize("invalid", ["X", "observations", "empty_observation", "empty_feature"])
    def test_malformed_schema(self, tmp_path: Path, invalid: str) -> None:
        write_iss_dataset(tmp_path)
        path = tmp_path / "cells.h5ad"
        table = ad.read_h5ad(path)
        if invalid == "X":
            table.X = None
        elif invalid == "observations":
            table.obs_names = ["7", "7"]
        elif invalid == "empty_observation":
            table.obs_names = ["", "2"]
        else:
            table.var_names = ["", "GeneA", "GeneB"]
        table.write_h5ad(path)
        with pytest.raises(ReaderFormatError):
            _read_table(ArtifactFile(path), instance_key=None, label_dtype=np.dtype(np.uint16))

    def test_feature_uniqueness_preserves_order_and_file(self, tmp_path: Path) -> None:
        write_iss_dataset(tmp_path)
        path = ArtifactFile(tmp_path / "cells.h5ad")
        table = _read_table(path, instance_key=None, label_dtype=np.dtype(np.uint16))
        assert table.var_names.tolist() == ["GeneB", "GeneA", "GeneB-1"]
        assert ad.read_h5ad(path).var_names.tolist() == ["GeneB", "GeneA", "GeneB"]


class TestSetInstanceIds:
    """Preserve exact integers and reject nonsensical label references."""

    @pytest.mark.parametrize("value", ["cell-a", "1.5", "0", "-1", "65536"])
    def test_invalid_index(self, tmp_path: Path, value: str) -> None:
        obs = pd.DataFrame(index=[value])
        with pytest.raises(ReaderFormatError):
            _set_instance_ids(
                obs, instance_key=None, label_dtype=np.dtype(np.uint16), path=ArtifactFile(tmp_path / "counts.h5ad")
            )

    @pytest.mark.parametrize("dtype", ["int8", "uint8", "Int64", "UInt64"])
    def test_integer_dtypes_and_nullable_storage(self, tmp_path: Path, dtype: str) -> None:
        obs = pd.DataFrame({"cell_id": pd.Series([7, 2], dtype=dtype)})
        _set_instance_ids(
            obs, instance_key="cell_id", label_dtype=np.dtype(np.uint16), path=ArtifactFile(tmp_path / "counts.h5ad")
        )
        np.testing.assert_array_equal(obs["cell_id"], [7, 2])
        assert obs["cell_id"].dtype == np.dtype(
            dtype.lower() if "64" in dtype else ("uint16" if "u" in dtype else "int16")
        )

    @pytest.mark.parametrize("values", [[True, False], [1.0, 2.0], ["1", "2"], [0, 2], [-1, 2], [65536, 2]])
    def test_invalid_explicit_values(
        self, tmp_path: Path, values: list[bool] | list[int] | list[float] | list[str]
    ) -> None:
        obs = pd.DataFrame({"cell_id": values})
        with pytest.raises(ReaderFormatError):
            _set_instance_ids(
                obs,
                instance_key="cell_id",
                label_dtype=np.dtype(np.uint16),
                path=ArtifactFile(tmp_path / "counts.h5ad"),
            )

    def test_null_integer_and_missing_column(self, tmp_path: Path) -> None:
        obs = pd.DataFrame({"cell_id": pd.Series([1, None], dtype="Int64")})
        for name in ("missing", "cell_id"):
            with pytest.raises(ReaderFormatError):
                _set_instance_ids(
                    obs, instance_key=name, label_dtype=np.dtype(np.uint16), path=ArtifactFile(tmp_path / "counts.h5ad")
                )

    @pytest.mark.parametrize("instance_key", [None, "cell_id"])
    def test_empty_ids_remain_typed(self, tmp_path: Path, instance_key: str | None) -> None:
        obs = pd.DataFrame({"cell_id": pd.Series([], dtype="int64")})
        _set_instance_ids(
            obs, instance_key=instance_key, label_dtype=np.dtype(np.uint16), path=ArtifactFile(tmp_path / "counts.h5ad")
        )
        assert obs[instance_key or "instance_id"].dtype == np.dtype(np.int64)
