"""Steinbock named-field flexibility and exact annotation schema failures."""

from __future__ import annotations

from decimal import localcontext
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.steinbock._schemas import (
    _centroid_columns,
    _format_error,
    _instance_identifiers,
    _metadata_columns,
    _validate_centroids,
    _validate_output_column,
    _validate_position_destinations,
    _validate_table,
    image_metadata_columns,
    normalize_image_metadata,
)
from tests.unit.readers.steinbock.test_tables import _sample

if TYPE_CHECKING:
    from pathlib import Path


class TestValidateTable:
    """Require named source columns with unrestricted order and preserved extras."""

    def test_reordered_fields_and_extensions(self, tmp_path: Path) -> None:
        """Reordered fields and extensions."""
        table = ad.AnnData(
            np.ones((1, 1)),
            obs=pd.DataFrame(
                {
                    "extra": pd.array([None], dtype="Int64"),
                    "centroid-1": [2.0],
                    "image": ["s.tiff"],
                    "centroid-0": [1.0],
                },
                index=["Object 1 in s.tiff"],
            ),
        )
        assert isinstance(table.obs, pd.DataFrame)
        before = table.obs.copy()
        _validate_table(table, ArtifactFile(tmp_path / "cells.h5ad"))
        pd.testing.assert_frame_equal(table.obs, before)

    @pytest.mark.parametrize("axis", ["obs", "var"])
    def test_duplicate_names(self, tmp_path: Path, axis: str) -> None:
        """Duplicate names."""
        table = ad.AnnData(np.ones((1, 1)))
        setattr(table, axis, pd.DataFrame([[1, 2]], columns=["same", "same"], index=["0"]))
        with pytest.raises(ReaderFormatError, match="duplicate annotation"):
            _validate_table(table, ArtifactFile(tmp_path / "cells.h5ad"))

    @pytest.mark.parametrize("axis", ["obs", "var"])
    def test_null_identifiers(self, tmp_path: Path, axis: str) -> None:
        """Reject nullable string index values before regex or linkage operations."""
        table = ad.AnnData(np.ones((1, 1)))
        frame = table.obs if axis == "obs" else table.var
        assert isinstance(frame, pd.DataFrame)
        frame.index = pd.Index(pd.array([None], dtype="string"))
        with pytest.raises(ReaderFormatError, match="identifiers"):
            _validate_table(table, ArtifactFile(tmp_path / "cells.h5ad"))


class TestValidateOutputColumn:
    """Never treat float-promoted or boolean IDs as equal to exact integers."""

    @pytest.mark.parametrize("existing", [2**63, float(2**63), None, True])
    def test_exact_collision(self, tmp_path: Path, existing: float | None) -> None:
        """Exact collision."""
        with pytest.raises(ReaderFormatError, match="conflicting reserved"):
            _validate_output_column(
                pd.DataFrame({"cell_id": [existing]}),
                "cell_id",
                pd.Series([2**63 + 1], dtype="uint64[pyarrow]"),
                path=ArtifactFile(tmp_path / "cells.h5ad"),
            )


class TestFormatError:
    """Retain table-specific reader and artifact identity in schema failures."""

    def test_source_context(self, tmp_path: Path) -> None:
        """Source context."""
        path = ArtifactFile(tmp_path / "cells.h5ad")

        error = _format_error(path, found="invalid rows", expected="linked cells")

        assert error.path == path
        assert error.context.reader == "steinbock"
        assert error.context.artifact == "cell table"
        assert "invalid rows" in str(error)


class TestInstanceIdentifiers:
    """Validate source schema invariants at their owning boundary."""

    @pytest.mark.parametrize(
        "observation",
        [
            "Object 0 in sample.tiff",
            "Object -1 in sample.tiff",
            "Object 1.5 in sample.tiff",
            "sample 2",
            "Object 2 in wrong.tiff",
            "Object 65536 in sample.tiff",
        ],
    )
    def test_malformed_or_out_of_range_ids(self, tmp_path: Path, observation: str) -> None:
        """Malformed or out of range ids."""
        obs = pd.DataFrame({"image": ["sample.tiff"]}, index=[observation])

        with pytest.raises(ReaderFormatError):
            _instance_identifiers(obs, {"sample": _sample()}, ArtifactFile(tmp_path / "cells.h5ad"))

    def test_dtype_bounds_are_sample_local(self, tmp_path: Path) -> None:
        """Dtype bounds are sample local."""
        obs = pd.DataFrame(
            {"image": ["small.tiff", "large.tiff"]}, index=["Object 256 in small.tiff", "Object 256 in large.tiff"]
        )

        with pytest.raises(ReaderFormatError, match=r"small\.tiff"):
            _instance_identifiers(
                obs,
                {"small": _sample(dtype="uint8"), "large": _sample(dtype="uint64")},
                ArtifactFile(tmp_path / "cells.h5ad"),
            )

    @pytest.mark.parametrize("images", [[None], ["other.tiff"], ["sample.tiff", "other.tiff"]])
    def test_null_unknown_or_missing_samples(self, tmp_path: Path, images: list[str | None]) -> None:
        """Null unknown or missing samples."""
        obs = pd.DataFrame(
            {"image": images}, index=[f"Object {number} in sample.tiff" for number in range(1, len(images) + 1)]
        )

        with pytest.raises(ReaderFormatError, match="sample disagreement"):
            _instance_identifiers(obs, {"sample": _sample()}, ArtifactFile(tmp_path / "cells.h5ad"))

    def test_missing_discovered_sample(self, tmp_path: Path) -> None:
        """Missing discovered sample."""
        obs = pd.DataFrame({"image": ["sample.tiff"]}, index=["Object 1 in sample.tiff"])

        with pytest.raises(ReaderFormatError, match="sample disagreement"):
            _instance_identifiers(obs, {"sample": _sample(), "other": _sample()}, ArtifactFile(tmp_path / "cells.h5ad"))


class TestValidateCentroids:
    """Validate source schema invariants at their owning boundary."""

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_invalid_validate_centroids(self, tmp_path: Path, value: float | str) -> None:
        """Invalid validate centroids."""
        obs = pd.DataFrame({"image": ["sample.tiff"], "centroid-0": [value], "centroid-1": [2.0]})

        with pytest.raises(ReaderFormatError, match="invalid centroids"):
            _validate_centroids(
                obs[["centroid-1", "centroid-0"]].to_numpy(),
                obs,
                {"sample": _sample()},
                ArtifactFile(tmp_path / "cells.h5ad"),
            )

    @pytest.mark.parametrize(("y", "x"), [(-0.1, 1), (1, -0.1), (8, 1), (1, 12)])
    def test_outside_validate_centroids(self, tmp_path: Path, y: float, x: float) -> None:
        """Outside validate centroids."""
        obs = pd.DataFrame({"image": ["sample.tiff"], "centroid-0": [y], "centroid-1": [x]})

        with pytest.raises(ReaderFormatError, match="outside"):
            _validate_centroids(
                obs[["centroid-1", "centroid-0"]].to_numpy(),
                obs,
                {"sample": _sample()},
                ArtifactFile(tmp_path / "cells.h5ad"),
            )

    def test_frame_bounds_are_sample_local(self, tmp_path: Path) -> None:
        """Frame bounds are sample local."""
        obs = pd.DataFrame({"image": ["small.tiff", "large.tiff"], "centroid-0": [5, 5], "centroid-1": [2, 2]})

        with pytest.raises(ReaderFormatError, match=r"small\.tiff"):
            _validate_centroids(
                obs[["centroid-1", "centroid-0"]].to_numpy(),
                obs,
                {"small": _sample(shape=(4, 4)), "large": _sample(shape=(8, 12))},
                ArtifactFile(tmp_path / "cells.h5ad"),
            )


class TestMetadataColumns:
    """Validate source schema invariants at their owning boundary."""

    @pytest.mark.parametrize("value", ["other.tiff", None])
    def test_conflicting_duplicate_image(self, tmp_path: Path, value: str | None) -> None:
        """Conflicting duplicate image."""
        obs = pd.DataFrame({"image": ["sample.tiff"], "Image": [value]})

        with pytest.raises(ReaderFormatError, match="conflicting"):
            _metadata_columns(obs, pd.DataFrame(), ArtifactFile(tmp_path / "cells.h5ad"))

    @pytest.mark.parametrize(
        "columns", [[""], [1], ["A B", "A_B"], ["Final Concentration / Dilution", "Final Concentration"]]
    )
    def test_invalid_or_colliding_feature_keys(self, tmp_path: Path, columns: list[str] | list[int]) -> None:
        """Invalid or colliding feature keys."""
        var = pd.DataFrame([[1] * len(columns)], columns=columns)

        with pytest.raises(ReaderFormatError):
            _metadata_columns(pd.DataFrame({"image": ["sample.tiff"]}), var, ArtifactFile(tmp_path / "cells.h5ad"))


class TestCentroidColumns:
    """Normalize supported nullable numerics while preserving exact source columns."""

    @pytest.mark.parametrize(
        "dtype", ["Int8", "UInt16", "Int64", "UInt64", "Float32", "Float64", "int64[pyarrow]", "float32[pyarrow]"]
    )
    def test_native_arrays_and_metadata_preservation(self, tmp_path: Path, dtype: str) -> None:
        """Extract each supported numeric representation without mutating obs."""
        obs = pd.DataFrame({"centroid-0": pd.Series([1, 3], dtype=dtype), "centroid-1": pd.Series([2, 4], dtype=dtype)})
        before = obs.copy()
        x, y = _centroid_columns(obs, ArtifactFile(tmp_path / "cells.h5ad"))
        np.testing.assert_array_equal(x, [2, 4])
        np.testing.assert_array_equal(y, [1, 3])
        assert x.dtype == y.dtype == np.dtype(dtype.lower().removesuffix("[pyarrow]"))
        pd.testing.assert_frame_equal(obs, before, check_exact=True)

    @pytest.mark.parametrize("dtype", ["Int64", "UInt64", "Float64", "boolean", "object", "string"])
    def test_null_boolean_and_nonnumeric_columns(self, tmp_path: Path, dtype: str) -> None:
        """Reject unsupported or missing centroid scalars before numerical assembly."""
        values = [True] if dtype == "boolean" else [None] if dtype in {"Int64", "UInt64", "Float64"} else ["1"]
        obs = pd.DataFrame({"centroid-0": pd.Series(values, dtype=dtype), "centroid-1": [2]})
        with pytest.raises(ReaderFormatError, match="invalid centroids"):
            _centroid_columns(obs, ArtifactFile(tmp_path / "cells.h5ad"))

    @pytest.mark.parametrize("value", [2**53 + 1, 2**63 + 1, 2**64 - 1])
    def test_lossy_integer_float_mixture(self, tmp_path: Path, value: int) -> None:
        """Reject integer coordinates that a common float64 dtype would round."""
        obs = pd.DataFrame({"centroid-0": pd.Series([value], dtype="UInt64"), "centroid-1": [1.0]})
        with pytest.raises(ReaderFormatError, match="centroid integer precision loss"):
            _centroid_columns(obs, ArtifactFile(tmp_path / "cells.h5ad"))

    @pytest.mark.parametrize("value", [2**53 + 2, -(2**53 + 2), -(2**63)])
    def test_exact_wide_integer_float_mixture(self, tmp_path: Path, value: int) -> None:
        """Accept exactly representable wide integers without an arbitrary magnitude ban."""
        obs = pd.DataFrame({"centroid-0": pd.Series([value], dtype="Int64"), "centroid-1": [1.0]})
        _, y = _centroid_columns(obs, ArtifactFile(tmp_path / "cells.h5ad"))
        assert int(y[0]) == value

    def test_uint64_columns_remain_exact(self, tmp_path: Path) -> None:
        """Keep homogeneous uint64 coordinates exact through native extraction."""
        obs = pd.DataFrame(
            {"centroid-0": pd.Series([2**64 - 1], dtype="UInt64"), "centroid-1": pd.Series([2**63 + 1], dtype="UInt64")}
        )
        x, y = _centroid_columns(obs, ArtifactFile(tmp_path / "cells.h5ad"))
        assert x.dtype == y.dtype == np.dtype(np.uint64)
        assert int(x[0]) == 2**63 + 1
        assert int(y[0]) == 2**64 - 1


class TestImageMetadataColumns:
    """Project named fields by source position and reject ambiguous header contracts."""

    def test_projects_reordered_named_fields(self, tmp_path: Path) -> None:
        positions = image_metadata_columns(["height_px", "extra", "image", "width_px"], ArtifactFile(tmp_path))
        assert positions == {"height_px": 0, "image": 2, "width_px": 3}

    @pytest.mark.parametrize(
        "header",
        [
            [],
            ["image", "image", "width_px", "height_px"],
            ["image", "width_px"],
            ["image", "width_px", "height_px", "acquisition_height_um"],
        ],
    )
    def test_invalid_headers(self, tmp_path: Path, header: list[str]) -> None:
        with pytest.raises(ReaderFormatError):
            image_metadata_columns(header, ArtifactFile(tmp_path))


class TestNormalizeImageMetadata:
    """Validate exact dimensions, complete extents and representable local lengths."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("width_px", "not-a-number"),
            ("width_px", "12.5"),
            ("height_px", "0"),
            ("width_px", str(2**63)),
            ("acquisition_width_um", ""),
            ("acquisition_width_um", "-1"),
            ("acquisition_width_um", "nan"),
            ("acquisition_width_um", "inf"),
            ("acquisition_width_um", "1e999"),
            ("acquisition_width_um", "1e-999"),
            ("acquisition_width_um", "5e-324"),
            ("acquisition_width_um", "1e999999999"),
            ("acquisition_width_um", "1e-999999999"),
        ],
    )
    def test_invalid_selected_values(self, tmp_path: Path, field: str, value: str) -> None:
        values = {
            "image": ["s.tiff"],
            "width_px": ["12"],
            "height_px": ["8"],
            "acquisition_width_um": ["6"],
            "acquisition_height_um": ["20"],
        }
        values[field] = [value]
        with pytest.raises(ReaderFormatError):
            normalize_image_metadata(values, {"s.tiff"}, ArtifactFile(tmp_path))

    @pytest.mark.parametrize("names", [[""], ["s.tiff", "s.tiff"]])
    def test_invalid_identities(self, tmp_path: Path, names: list[str]) -> None:
        with pytest.raises(ReaderFormatError, match="image names"):
            normalize_image_metadata({"image": names}, {"s.tiff"}, ArtifactFile(tmp_path))

    def test_ratios_are_independent_of_decimal_context(self, tmp_path: Path) -> None:
        columns = {
            "image": ["s.tiff"],
            "width_px": ["11"],
            "height_px": ["13"],
            "acquisition_width_um": ["3"],
            "acquisition_height_um": ["7"],
        }
        with localcontext() as context:
            context.prec = 2
            metadata = normalize_image_metadata(columns, {"s.tiff"}, ArtifactFile(tmp_path))
        assert metadata["s.tiff"].spacing_um == (3 / 11, 7 / 13)


class TestValidatePositionDestinations:
    @pytest.mark.parametrize(
        "collision", ["spatial_pixels", "spatial_source", "declaration", "spatial_frame", "spatial_pixel_frame"]
    )
    def test_existing_destinations_fail_before_mutation(self, tmp_path: Path, collision: str) -> None:
        table = ad.AnnData(np.ones((2, 1)))
        if collision == "declaration":
            table.uns[POSITIONS_KEY] = {"schema_version": 1}
        elif collision in {"spatial_frame", "spatial_pixel_frame"}:
            table.obs[collision] = pd.Categorical(["s", "s"])
        else:
            table.obsm[collision] = np.zeros((2, 3), dtype=np.int16)
            if collision == "spatial_source":
                table.obsm["spatial"] = np.zeros((2, 3), dtype=np.int16)
        with pytest.raises(ReaderFormatError, match="existing position"):
            _validate_position_destinations(table, multiple=True, path=ArtifactFile(tmp_path / "cells.h5ad"))
        assert "cell_id" not in table.obs

    def test_unrelated_source_embedding_is_retained(self, tmp_path: Path) -> None:
        table = ad.AnnData(np.ones((1, 1)))
        table.obsm["spatial_source"] = np.zeros((1, 3), dtype=np.int16)
        _validate_position_destinations(table, multiple=False, path=ArtifactFile(tmp_path / "cells.h5ad"))
        assert "spatial_source" in table.obsm
