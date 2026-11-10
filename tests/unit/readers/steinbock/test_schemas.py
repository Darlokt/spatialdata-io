"""Steinbock named-field flexibility and exact annotation schema failures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.steinbock._images import SampleGrid
from spatialdata_io.readers.steinbock._schemas import (
    IMAGE_METADATA_TYPES,
    _format_error,
    centroid_columns,
    image_metadata_types,
    instance_identifiers,
    metadata_columns,
    normalize_image_metadata,
    validate_centroids,
    validate_position_destinations,
    validate_table,
)

if TYPE_CHECKING:
    from pathlib import Path


def _grid(name: str, shape: tuple[int, int], dtype: np.dtype[np.integer]) -> SampleGrid:
    """Describe an uncalibrated mask grid without reading rasters."""
    return SampleGrid(f"{name}.tiff", shape, dtype, "sample", None)


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
        validate_table(table, ArtifactFile(tmp_path / "cells.h5ad"))
        pd.testing.assert_frame_equal(table.obs, before)

    @pytest.mark.parametrize("axis", ["obs", "var"])
    def test_duplicate_names(self, tmp_path: Path, axis: str) -> None:
        """Duplicate names."""
        table = ad.AnnData(np.ones((1, 1)))
        setattr(table, axis, pd.DataFrame([[1, 2]], columns=["same", "same"], index=["0"]))
        with pytest.raises(ReaderFormatError, match="duplicate annotation"):
            validate_table(table, ArtifactFile(tmp_path / "cells.h5ad"))

    @pytest.mark.parametrize("axis", ["obs", "var"])
    def test_null_identifiers(self, tmp_path: Path, axis: str) -> None:
        """Reject nullable string index values before regex or linkage operations."""
        table = ad.AnnData(np.ones((1, 1)))
        frame = table.obs if axis == "obs" else table.var
        assert isinstance(frame, pd.DataFrame)
        frame.index = pd.Index(pd.array([None], dtype="string"))
        with pytest.raises(ReaderFormatError, match="identifiers"):
            validate_table(table, ArtifactFile(tmp_path / "cells.h5ad"))


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

    @pytest.mark.parametrize("identifier", [0, 2**64])
    def test_uint64_background_and_overflow_are_rejected(self, tmp_path: Path, identifier: int) -> None:
        obs = pd.DataFrame({"image": ["sample.tiff"]}, index=[f"Object {identifier} in sample.tiff"])

        with pytest.raises(ReaderFormatError, match="positive ID fitting its mask dtype"):
            instance_identifiers(
                obs, {"sample": _grid("sample", (8, 12), np.dtype("uint64"))}, ArtifactFile(tmp_path / "cells.h5ad")
            )

    @pytest.mark.parametrize(
        "observation",
        [
            "Object 0 in sample.tiff",
            "Object -1 in sample.tiff",
            "Object 1.5 in sample.tiff",
            "sample 2",
            "Object 2 in wrong.tiff",
            "Object 65536 in sample.tiff",
            "Object 18446744073709551616 in sample.tiff",
        ],
    )
    def test_malformed_or_out_of_range_ids(self, tmp_path: Path, observation: str) -> None:
        """Malformed or out of range ids."""
        obs = pd.DataFrame({"image": ["sample.tiff"]}, index=[observation])

        with pytest.raises(ReaderFormatError):
            instance_identifiers(
                obs, {"sample": _grid("sample", (8, 12), np.dtype("uint16"))}, ArtifactFile(tmp_path / "cells.h5ad")
            )

    def test_dtype_bounds_are_sample_local(self, tmp_path: Path) -> None:
        """Dtype bounds are sample local."""
        obs = pd.DataFrame(
            {"image": ["small.tiff", "large.tiff"]}, index=["Object 256 in small.tiff", "Object 256 in large.tiff"]
        )

        with pytest.raises(ReaderFormatError, match=r"small\.tiff"):
            instance_identifiers(
                obs,
                {
                    "small": _grid("small", (8, 12), np.dtype("uint8")),
                    "large": _grid("large", (8, 12), np.dtype("uint64")),
                },
                ArtifactFile(tmp_path / "cells.h5ad"),
            )

    @pytest.mark.parametrize("images", [[None], ["other.tiff"], ["sample.tiff", "other.tiff"]])
    def test_null_unknown_or_missing_samples(self, tmp_path: Path, images: list[str | None]) -> None:
        """Null unknown or missing samples."""
        obs = pd.DataFrame(
            {"image": images}, index=[f"Object {number} in sample.tiff" for number in range(1, len(images) + 1)]
        )

        with pytest.raises(ReaderFormatError, match="sample disagreement"):
            instance_identifiers(
                obs, {"sample": _grid("sample", (8, 12), np.dtype("uint16"))}, ArtifactFile(tmp_path / "cells.h5ad")
            )

    def test_missing_discovered_sample(self, tmp_path: Path) -> None:
        """Missing discovered sample."""
        obs = pd.DataFrame({"image": ["sample.tiff"]}, index=["Object 1 in sample.tiff"])

        with pytest.raises(ReaderFormatError, match="sample disagreement"):
            instance_identifiers(
                obs,
                {
                    "sample": _grid("sample", (8, 12), np.dtype("uint16")),
                    "other": _grid("other", (8, 12), np.dtype("uint16")),
                },
                ArtifactFile(tmp_path / "cells.h5ad"),
            )


class TestValidateCentroids:
    """Validate source schema invariants at their owning boundary."""

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_invalid_validate_centroids(self, tmp_path: Path, value: float | str) -> None:
        """Invalid validate centroids."""
        obs = pd.DataFrame({"image": ["sample.tiff"], "centroid-0": [value], "centroid-1": [2.0]})

        with pytest.raises(ReaderFormatError, match="invalid centroids"):
            validate_centroids(
                obs[["centroid-1", "centroid-0"]].to_numpy(),
                obs,
                {"sample": _grid("sample", (8, 12), np.dtype("uint16"))},
                ArtifactFile(tmp_path / "cells.h5ad"),
            )

    @pytest.mark.parametrize(("y", "x"), [(-0.1, 1), (1, -0.1), (8, 1), (1, 12)])
    def test_outside_validate_centroids(self, tmp_path: Path, y: float, x: float) -> None:
        """Outside validate centroids."""
        obs = pd.DataFrame({"image": ["sample.tiff"], "centroid-0": [y], "centroid-1": [x]})

        with pytest.raises(ReaderFormatError, match="outside"):
            validate_centroids(
                obs[["centroid-1", "centroid-0"]].to_numpy(),
                obs,
                {"sample": _grid("sample", (8, 12), np.dtype("uint16"))},
                ArtifactFile(tmp_path / "cells.h5ad"),
            )

    def test_frame_bounds_are_sample_local(self, tmp_path: Path) -> None:
        """Frame bounds are sample local."""
        obs = pd.DataFrame({"image": ["small.tiff", "large.tiff"], "centroid-0": [5, 5], "centroid-1": [2, 2]})

        with pytest.raises(ReaderFormatError, match=r"small\.tiff"):
            validate_centroids(
                obs[["centroid-1", "centroid-0"]].to_numpy(),
                obs,
                {
                    "small": _grid("small", (4, 4), np.dtype("uint16")),
                    "large": _grid("large", (8, 12), np.dtype("uint16")),
                },
                ArtifactFile(tmp_path / "cells.h5ad"),
            )


class TestMetadataColumns:
    """Validate source schema invariants at their owning boundary."""

    @pytest.mark.parametrize("value", ["other.tiff", None])
    def test_conflicting_duplicate_image(self, tmp_path: Path, value: str | None) -> None:
        """Conflicting duplicate image."""
        obs = pd.DataFrame({"image": ["sample.tiff"], "Image": [value]})

        with pytest.raises(ReaderFormatError, match="conflicting"):
            metadata_columns(obs, pd.DataFrame(), ArtifactFile(tmp_path / "cells.h5ad"))

    @pytest.mark.parametrize("columns", [[""], [1], ["Final Concentration / Dilution", "Final_Concentration"]])
    def test_invalid_or_colliding_feature_keys(self, tmp_path: Path, columns: list[str] | list[int]) -> None:
        """Invalid or colliding feature keys."""
        var = pd.DataFrame([[1] * len(columns)], columns=columns)

        with pytest.raises(ReaderFormatError):
            metadata_columns(pd.DataFrame({"image": ["sample.tiff"]}), var, ArtifactFile(tmp_path / "cells.h5ad"))

    def test_only_the_legacy_dilution_name_is_renamed(self, tmp_path: Path) -> None:
        """Other names reach the shared column-name sanitizer unchanged."""
        columns = ["Channel Name", "Final Concentration / Dilution", "A B", "A_B"]
        var = pd.DataFrame([[1] * len(columns)], columns=columns)

        result = metadata_columns(pd.DataFrame({"image": ["sample.tiff"]}), var, ArtifactFile(tmp_path / "cells.h5ad"))

        assert result == ["Channel Name", "Final_Concentration", "A B", "A_B"]


class TestCentroidColumns:
    """Normalize supported nullable numerics while preserving exact source columns."""

    @pytest.mark.parametrize(
        "dtype", ["Int8", "UInt16", "Int64", "UInt64", "Float32", "Float64", "int64[pyarrow]", "float32[pyarrow]"]
    )
    def test_native_arrays_and_metadata_preservation(self, tmp_path: Path, dtype: str) -> None:
        """Extract each supported numeric representation without mutating obs."""
        obs = pd.DataFrame({"centroid-0": pd.Series([1, 3], dtype=dtype), "centroid-1": pd.Series([2, 4], dtype=dtype)})
        before = obs.copy()
        x, y = centroid_columns(obs, ArtifactFile(tmp_path / "cells.h5ad"))
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
            centroid_columns(obs, ArtifactFile(tmp_path / "cells.h5ad"))

    @pytest.mark.parametrize("value", [2**53 + 1, 2**63 + 1, 2**64 - 1])
    def test_lossy_integer_float_mixture(self, tmp_path: Path, value: int) -> None:
        """Reject integer coordinates that a common float64 dtype would round."""
        obs = pd.DataFrame({"centroid-0": pd.Series([value], dtype="UInt64"), "centroid-1": [1.0]})
        with pytest.raises(ReaderFormatError, match="centroid integer precision loss"):
            centroid_columns(obs, ArtifactFile(tmp_path / "cells.h5ad"))

    @pytest.mark.parametrize("value", [2**53 + 2, -(2**53 + 2), -(2**63)])
    def test_exact_wide_integer_float_mixture(self, tmp_path: Path, value: int) -> None:
        """Accept exactly representable wide integers without an arbitrary magnitude ban."""
        obs = pd.DataFrame({"centroid-0": pd.Series([value], dtype="Int64"), "centroid-1": [1.0]})
        _, y = centroid_columns(obs, ArtifactFile(tmp_path / "cells.h5ad"))
        assert int(y[0]) == value

    def test_uint64_columns_remain_exact(self, tmp_path: Path) -> None:
        """Keep homogeneous uint64 coordinates exact through native extraction."""
        obs = pd.DataFrame(
            {"centroid-0": pd.Series([2**64 - 1], dtype="UInt64"), "centroid-1": pd.Series([2**63 + 1], dtype="UInt64")}
        )
        x, y = centroid_columns(obs, ArtifactFile(tmp_path / "cells.h5ad"))
        assert x.dtype == y.dtype == np.dtype(np.uint64)
        assert int(x[0]) == 2**63 + 1
        assert int(y[0]) == 2**64 - 1


class TestImageMetadataTypes:
    """Project named fields by source position and reject ambiguous header contracts."""

    def test_projects_reordered_named_fields(self, tmp_path: Path) -> None:
        types = image_metadata_types(["height_px", "extra", "image", "width_px"], ArtifactFile(tmp_path))
        assert types == pa.schema([("height_px", pa.uint64()), ("image", pa.string()), ("width_px", pa.uint64())])

    @pytest.mark.parametrize(
        "header",
        [
            [],
            ["image", "width_px", "height_px", "ignored\x00name"],
            ["image", "width_px"],
            ["image", "width_px", "height_px", "acquisition_height_um"],
        ],
    )
    def test_invalid_headers(self, tmp_path: Path, header: list[str]) -> None:
        with pytest.raises(ReaderFormatError):
            image_metadata_types(header, ArtifactFile(tmp_path))


def _image_table(**columns: list[object]) -> pa.Table:
    """Return an images.csv table with the reader's Arrow types."""
    return pa.table({name: pa.array(values, type=IMAGE_METADATA_TYPES[name]) for name, values in columns.items()})


class TestNormalizeImageMetadata:
    """Validate positive dimensions, complete extents and representable spacing of selected images."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("width_px", None),
            ("height_px", 0),
            ("acquisition_width_um", None),
            ("acquisition_width_um", -1.0),
            ("acquisition_width_um", 0.0),
            ("acquisition_width_um", float("nan")),
            ("acquisition_width_um", float("inf")),
            ("acquisition_width_um", 5e-324),
        ],
    )
    def test_invalid_selected_values(self, tmp_path: Path, field: str, value: float | None) -> None:
        values: dict[str, list[object]] = {
            "image": ["s.tiff"],
            "width_px": [12],
            "height_px": [8],
            "acquisition_width_um": [6.0],
            "acquisition_height_um": [20.0],
        }
        values[field] = [value]
        with pytest.raises(ReaderFormatError, match="selected image metadata"):
            normalize_image_metadata(_image_table(**values), {"s.tiff"}, ArtifactFile(tmp_path))

    @pytest.mark.parametrize("names", [[""], ["s.tiff", "s.tiff"], ["s\x00.tiff"]])
    def test_invalid_identities(self, tmp_path: Path, names: list[object]) -> None:
        with pytest.raises(ReaderFormatError, match="image names"):
            normalize_image_metadata(
                _image_table(image=names, width_px=[1] * len(names), height_px=[1] * len(names)),
                set(),
                ArtifactFile(tmp_path),
            )

    def test_spacing_divides_float64_extents(self, tmp_path: Path) -> None:
        """Spacing is the float64 extent divided by the pixel count, in selected source row order."""
        table = _image_table(
            image=["unused.tiff", "b.tiff", "a.tiff"],
            width_px=[0, 7, 9],
            height_px=[0, 3, 4],
            acquisition_width_um=[None, 0.1, None],
            acquisition_height_um=[-1.0, 0.3, None],
        )

        metadata = normalize_image_metadata(table, {"a.tiff", "b.tiff"}, ArtifactFile(tmp_path))

        assert list(metadata) == ["b.tiff", "a.tiff"]
        assert metadata["b.tiff"].shape == (3, 7)
        assert metadata["b.tiff"].spacing_um == (0.1 / 7, 0.3 / 3)
        assert metadata["a.tiff"].shape == (4, 9)
        assert metadata["a.tiff"].spacing_um is None


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
            validate_position_destinations(table, multiple=True, path=ArtifactFile(tmp_path / "cells.h5ad"))
        assert "cell_id" not in table.obs

    def test_unrelated_source_embedding_is_retained(self, tmp_path: Path) -> None:
        table = ad.AnnData(np.ones((1, 1)))
        table.obsm["spatial_source"] = np.zeros((1, 3), dtype=np.int16)
        validate_position_destinations(table, multiple=False, path=ArtifactFile(tmp_path / "cells.h5ad"))
        assert "spatial_source" in table.obsm
