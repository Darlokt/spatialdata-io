"""ISS h5ad ownership, schema, and integer instance semantics."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatialdata_io.experimental.iss._tables import _set_instance_ids, read_table
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from tests.integration.experimental.iss.conftest import write_iss_dataset

if TYPE_CHECKING:
    from pathlib import Path


class TestReadTable:
    """Reject malformed scientific inputs before final linkage."""

    @pytest.mark.parametrize("present", [True, False])
    def test_unknown_embedding_preserves_shape_dtype_and_has_no_description(
        self, tmp_path: Path, *, present: bool
    ) -> None:
        write_iss_dataset(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        native = np.arange(6, dtype=np.int16).reshape(2, 3)
        if present:
            source.obsm["spatial"] = native
        else:
            source.obsm.pop("spatial", None)
        source.write_h5ad(path)
        table = read_table(ArtifactFile(path), region="cells", instance_key=None, label_dtype=np.dtype(np.uint16))
        assert "spatial" not in table.obsm
        assert "spatialdata_io_positions" not in table.uns
        if present:
            np.testing.assert_array_equal(table.obsm["spatial_source"], native)
            assert table.obsm["spatial_source"].dtype == np.int16
        else:
            assert "spatial_source" not in table.obsm

    def test_imported_destination_collision(self, tmp_path: Path) -> None:
        write_iss_dataset(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        source.obsm["spatial_source"] = np.ones((2, 4))
        source.write_h5ad(path)
        with pytest.raises(ReaderFormatError, match="spatial_source"):
            read_table(ArtifactFile(path), region="cells", instance_key=None, label_dtype=np.dtype(np.uint16))

    def test_uns_keys_are_sanitized_with_one_warning(self, tmp_path: Path) -> None:
        """Rename invalid and reserved-colliding top-level keys only, keeping values and stored order."""
        write_iss_dataset(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        source.uns["my key"] = {"inner key": "kept"}
        source.uns["valid"] = "v"
        source.uns["SpatialData_Attrs"] = np.asarray([1, 2])
        source.write_h5ad(path)

        with pytest.warns(UserWarning, match="uns keys") as caught:
            table = read_table(ArtifactFile(path), region="cells", instance_key=None, label_dtype=np.dtype(np.uint16))

        assert [str(warning.message) for warning in caught] == [
            (
                "iss: the H5AD uns has names that a SpatialData table does not accept as uns keys; using "
                "'SpatialData_Attrs' -> 'SpatialData_Attrs_1', 'my key' -> 'my_key'. Rename them in the source to "
                "choose the names explicitly."
            )
        ]
        # HDF5 stores group members by name, so the stored order is alphabetical.
        assert list(table.uns) == ["SpatialData_Attrs_1", "my_key", "valid"]
        assert table.uns["my_key"] == {"inner key": "kept"}
        assert table.uns["valid"] == "v"
        np.testing.assert_array_equal(table.uns["SpatialData_Attrs_1"], [1, 2])

    def test_valid_uns_keys_are_untouched_without_warning(self, tmp_path: Path) -> None:
        write_iss_dataset(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        source.uns["qc"] = {"inner key": 1}
        source.write_h5ad(path)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            table = read_table(ArtifactFile(path), region="cells", instance_key=None, label_dtype=np.dtype(np.uint16))

        assert list(table.uns) == ["qc"]
        assert table.uns["qc"] == {"inner key": 1}

    def test_read_failure_has_context(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.h5ad"
        path.write_bytes(b"not HDF5")

        with pytest.raises(OSError, match="file signature") as caught:
            read_table(ArtifactFile(path), region="cells", instance_key=None, label_dtype=np.dtype(np.uint16))

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
            read_table(ArtifactFile(path), region="cells", instance_key=None, label_dtype=np.dtype(np.uint16))

    @pytest.mark.parametrize("instance_key", [None, "segmentation_id"])
    def test_linkage_columns_and_annotation_storage(self, tmp_path: Path, instance_key: str | None) -> None:
        write_iss_dataset(tmp_path)

        table = read_table(
            ArtifactFile(tmp_path / "cells.h5ad"),
            region="fov_labels_image",
            instance_key=instance_key,
            label_dtype=np.dtype(np.uint16),
        )

        assert isinstance(table.obs, pd.DataFrame)
        expected = ["segmentation_id", "cell_type"] + ([] if instance_key else ["instance_id"]) + ["region"]
        assert table.obs.columns.tolist() == expected
        assert table.obs["region"].tolist() == ["fov_labels_image"] * 2
        assert isinstance(table.obs["region"].dtype, pd.CategoricalDtype)
        assert table.obs[instance_key or "instance_id"].tolist() == [7, 2]
        # Arrow-backed generated IDs are converted to AnnData-compatible storage before linkage.
        assert str(table.obs[instance_key or "instance_id"].dtype) == "int64"

    def test_conflicting_region_field(self, tmp_path: Path) -> None:
        write_iss_dataset(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        source.obs["region"] = ["fov_labels_image", "other"]
        source.write_h5ad(path)

        with pytest.raises(ReaderFormatError, match="conflicting reserved observation column 'region'"):
            read_table(
                ArtifactFile(path), region="fov_labels_image", instance_key=None, label_dtype=np.dtype(np.uint16)
            )

    def test_duplicate_feature_warning_is_suppressed_only_inside_the_read(
        self, tmp_path: Path, recwarn: pytest.WarningsRecorder
    ) -> None:
        write_iss_dataset(tmp_path)
        # Control: reading the same file directly emits AnnData's advice.
        with pytest.warns(UserWarning, match=r"Variable names are not unique"):
            ad.read_h5ad(tmp_path / "cells.h5ad")
        recwarn.clear()
        filters = list(warnings.filters)

        table = read_table(
            ArtifactFile(tmp_path / "cells.h5ad"), region="cells", instance_key=None, label_dtype=np.dtype(np.uint16)
        )

        assert table.var_names.tolist() == ["GeneB", "GeneA", "GeneB-1"]
        assert not [warning for warning in recwarn if "names are not unique" in str(warning.message)]
        assert warnings.filters == filters

    def test_other_warnings_of_the_read_still_reach_the_caller(self, tmp_path: Path) -> None:
        write_iss_dataset(tmp_path)
        path = tmp_path / "cells.h5ad"
        source = ad.read_h5ad(path)
        source.obs_names = ["7", "7"]
        source.write_h5ad(path)
        filters = list(warnings.filters)

        with (
            pytest.raises(ReaderFormatError, match="duplicate observations"),
            pytest.warns(UserWarning, match=r"Observation names are not unique") as reported,
        ):
            read_table(ArtifactFile(path), region="cells", instance_key=None, label_dtype=np.dtype(np.uint16))

        assert not [warning for warning in reported if "Variable names" in str(warning.message)]
        assert warnings.filters == filters

    def test_feature_uniqueness_preserves_order_and_file(self, tmp_path: Path) -> None:
        write_iss_dataset(tmp_path)
        path = ArtifactFile(tmp_path / "cells.h5ad")

        table = read_table(path, region="cells", instance_key=None, label_dtype=np.dtype(np.uint16))

        assert table.var_names.tolist() == ["GeneB", "GeneA", "GeneB-1"]
        assert ad.read_h5ad(path).var_names.tolist() == ["GeneB", "GeneA", "GeneB"]


class TestSetInstanceIds:
    """Preserve exact integers and reject nonsensical label references."""

    @pytest.mark.parametrize("value", ["cell-a", "1.5", "0_7", "+2", "-2", " 2", "2 ", "\u0667", "", "1e3"])
    def test_names_other_than_ascii_digits_are_rejected(self, tmp_path: Path, value: str) -> None:
        # int() would accept several of these spellings and link the row to a wrong or unintended label.
        obs = pd.DataFrame(index=["3", value])

        with pytest.raises(ReaderFormatError, match="not ASCII decimal digits"):
            _set_instance_ids(
                obs, instance_key=None, label_dtype=np.dtype(np.uint16), path=ArtifactFile(tmp_path / "counts.h5ad")
            )

    @pytest.mark.parametrize(
        "dtype",
        [
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "Int64",
            "UInt64",
            "uint64[pyarrow]",
        ],
    )
    def test_integer_dtypes_and_nullable_storage(self, tmp_path: Path, dtype: str) -> None:
        """Integer dtypes and nullable storage."""
        obs = pd.DataFrame({"cell_id": pd.Series([7, 2], dtype=dtype)})

        _set_instance_ids(
            obs, instance_key="cell_id", label_dtype=np.dtype(np.uint16), path=ArtifactFile(tmp_path / "counts.h5ad")
        )
        np.testing.assert_array_equal(obs["cell_id"], [7, 2])
        if dtype in {"int8", "uint8"}:
            expected = "uint16[pyarrow]" if dtype == "uint8" else "int16[pyarrow]"
        elif dtype in {"Int64", "UInt64", "uint64[pyarrow]"}:
            expected = f"{dtype.lower().removesuffix('[pyarrow]')}[pyarrow]"
        else:
            expected = dtype
        assert str(obs["cell_id"].dtype) == expected

    @pytest.mark.parametrize("instance_key", [None, "cell_id"])
    def test_empty_ids_remain_typed(self, tmp_path: Path, instance_key: str | None) -> None:
        """Empty ids remain typed."""
        obs = pd.DataFrame({"cell_id": pd.Series([], dtype="int64")})

        _set_instance_ids(
            obs, instance_key=instance_key, label_dtype=np.dtype(np.uint16), path=ArtifactFile(tmp_path / "counts.h5ad")
        )

        assert str(obs[instance_key or "instance_id"].dtype) == ("int64[pyarrow]" if instance_key is None else "int64")

    @pytest.mark.parametrize("dtype", ["int16", "uint16", "int32", "uint32", "int64", "uint64"])
    @pytest.mark.parametrize("empty", [False, True])
    def test_native_ids_preserve_column_storage_and_order(self, tmp_path: Path, dtype: str, *, empty: bool) -> None:
        index = pd.Index([] if empty else ["cell-b", "cell-a"], name="cells")
        obs = pd.DataFrame(
            {
                "before": "annotation",
                "cell_id": pd.Series([] if empty else [7, 2], index=index, dtype=dtype),
                "after": 3,
            },
            index=index,
        )
        original = obs.copy()
        native = obs["cell_id"].to_numpy()
        _set_instance_ids(
            obs, instance_key="cell_id", label_dtype=np.dtype(np.uint64), path=ArtifactFile(tmp_path / "counts.h5ad")
        )
        pd.testing.assert_frame_equal(obs, original)
        if not empty:
            assert np.shares_memory(obs["cell_id"].to_numpy(), native)

    @pytest.mark.parametrize("dtype", ["uint64", "UInt64", "uint64[pyarrow]"])
    def test_wide_explicit_ids_remain_exact(self, tmp_path: Path, dtype: str) -> None:
        obs = pd.DataFrame({"cell_id": pd.Series([2**63 + 1, 2**64 - 1], dtype=dtype)})
        _set_instance_ids(
            obs, instance_key="cell_id", label_dtype=np.dtype(np.uint64), path=ArtifactFile(tmp_path / "counts.h5ad")
        )
        assert obs["cell_id"].tolist() == [2**63 + 1, 2**64 - 1]
        assert str(obs["cell_id"].dtype) == ("uint64" if dtype == "uint64" else "uint64[pyarrow]")
