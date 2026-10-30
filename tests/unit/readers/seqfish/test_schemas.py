"""seqFISH field contracts, exact IDs, finite coordinates, and lazy point partitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.seqfish._schemas import (
    FLOAT_DTYPE,
    STRING_DTYPE,
    boundary_features,
    normalize_floats,
    normalize_ids,
    normalize_transcript_partition,
    validate_cell_columns,
    validate_count_columns,
    validate_transcript_columns,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestValidateCountColumns:
    """Preserve positional identity and exact feature spelling and order."""

    @pytest.mark.parametrize("first", ["", "label"])
    def test_positional_identity(self, tmp_path: Path, first: str) -> None:
        assert validate_count_columns([first, "Gene-B", "GeneA"], ArtifactFile(tmp_path / "counts.csv")) == (
            "Gene-B",
            "GeneA",
        )

    @pytest.mark.parametrize("columns", [[], [""], ["", ""], ["", "A", "A"], ["", "A\x00"]])
    def test_ambiguous_headers(self, tmp_path: Path, columns: list[str]) -> None:
        with pytest.raises(ReaderFormatError):
            validate_count_columns(columns, ArtifactFile(tmp_path / "counts.csv"))


class TestValidateCellColumns:
    """Project named metadata independent of input column order."""

    def test_order_and_extras(self, tmp_path: Path) -> None:
        assert validate_cell_columns(
            ["extra", "center_y", "area", "label", "center_x"], ArtifactFile(tmp_path / "cells.csv")
        ) == ("label", "area", "center_x", "center_y")

    def test_missing_fields(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="cell fields"):
            validate_cell_columns(["label", "area"], ArtifactFile(tmp_path / "cells.csv"))


class TestNormalizeIds:
    """Retain exact identity across accepted lexical integer representations."""

    def test_exact_large_and_decimal_ids(self, tmp_path: Path) -> None:
        result = normalize_ids(pd.Series([str(2**53 + 1), "02.0", "3e0"]), ArtifactFile(tmp_path / "cells.csv"))
        np.testing.assert_array_equal(result, [2**53 + 1, 2, 3])
        assert result.dtype == np.dtype(np.int64)

    def test_unsigned_and_empty(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "cells.csv")
        assert normalize_ids(pd.Series([str(2**64 - 1)]), path).dtype == np.dtype(np.uint64)
        assert normalize_ids(pd.Series([], dtype=STRING_DTYPE), path).dtype == np.dtype(np.int64)
        assert normalize_ids(pd.Series([float(2**63)]), path).tolist() == [2**63]
        assert normalize_ids(pd.Series(["0"]), path, allow_zero=True).tolist() == [0]

    def test_arrow_zero_identifier(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="positive cell"):
            normalize_ids(pd.Series(["0"], dtype=STRING_DTYPE), ArtifactFile(tmp_path / "cells.csv"))

    @pytest.mark.parametrize("storage", ["pyarrow", "python"])
    def test_pandas_string_identifiers(self, tmp_path: Path, storage: Literal["pyarrow", "python"]) -> None:
        result = normalize_ids(
            pd.Series(["0", str(2**63 + 1)], dtype=pd.StringDtype(storage=storage)),
            ArtifactFile(tmp_path / "tx.csv"),
            allow_zero=True,
        )
        assert result.dtype == np.dtype(np.uint64)
        assert result.tolist() == [0, 2**63 + 1]

    @pytest.mark.parametrize(
        "value", [None, True, "", "NaN", "inf", "0", "-1", "1.00000000000000001", "1e10000", str(2**64), "1" * 129]
    )
    def test_invalid_identifiers(self, tmp_path: Path, *, value: str | bool | None) -> None:
        with pytest.raises(ReaderFormatError, match="identifier"):
            normalize_ids(pd.Series([value]), ArtifactFile(tmp_path / "cells.csv"))


class TestNormalizeFloats:
    """Keep correctly rounded precision while rejecting destructive conversion."""

    def test_round_trip_and_index(self, tmp_path: Path) -> None:
        result = normalize_floats(
            pd.Series(["0.10000000000000002", "-0.00e1000", "9007199254740992"], index=[2, 4, 8]),
            ArtifactFile(tmp_path / "cells.csv"),
            "x",
        )
        assert result.dtype == FLOAT_DTYPE
        assert result.index.tolist() == [2, 4, 8]
        np.testing.assert_array_equal(result.to_numpy(dtype=np.float64), [0.10000000000000002, 0, 2**53])

    @pytest.mark.parametrize("value", [None, True, "", "nan", "inf", "1e-400", "1e400", "9007199254740993", "1" * 129])
    def test_invalid_coordinates(self, tmp_path: Path, *, value: str | bool | None) -> None:
        with pytest.raises(ReaderFormatError, match="float64"):
            normalize_floats(pd.Series([value]), ArtifactFile(tmp_path / "cells.csv"), "x")


class TestValidateTranscriptColumns:
    """Declare optional plane and assignment fields without accepting repaired names."""

    def test_projection(self, tmp_path: Path) -> None:
        assert validate_transcript_columns(
            ["extra", "cell", "name", "y", "z", "x"], ArtifactFile(tmp_path / "tx.csv")
        ) == ("name", "x", "y", "z", "cell")

    def test_required(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="transcript fields"):
            validate_transcript_columns(["x", "y"], ArtifactFile(tmp_path / "tx.csv"))


class TestNormalizeTranscriptPartition:
    """Normalize only the executed partition and preserve unassigned events."""

    def test_partition_values_and_empty(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "tx.csv")
        frame = pd.DataFrame(
            {"name": ["B", "A"], "x": ["0.1", "2"], "y": ["3", "4"], "z": ["0", "2"], "cell": ["0", str(2**63)]},
            dtype=STRING_DTYPE,
        )
        result = normalize_transcript_partition(frame, features=("A", "B", "ZeroCount"), path=path)
        assert result["name"].cat.categories.tolist() == ["A", "B", "ZeroCount"]
        assert result["name"].tolist() == ["B", "A"]
        assert result["x"].dtype == np.dtype(np.float64)
        assert result["cell"].tolist() == [0, 2**63]
        assert result["z_raw"].tolist() == [0, 2]
        assert "z" not in result
        assert normalize_transcript_partition(
            result.iloc[:0].drop(columns="z_raw"), features=("A", "B"), path=path
        ).empty

    @pytest.mark.parametrize("name", ["unknown", None])
    def test_unknown_features(self, tmp_path: Path, name: str | None) -> None:
        with pytest.raises(ReaderFormatError, match="vocabulary"):
            normalize_transcript_partition(
                pd.DataFrame({"name": [name]}), features=("A",), path=ArtifactFile(tmp_path / "tx.csv")
            )

    def test_plane_overflow(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"name": ["A"], "x": ["1"], "y": ["1"], "z": [str(2**63)]})
        with pytest.raises(ReaderFormatError, match="plane overflow"):
            normalize_transcript_partition(frame, features=("A",), path=ArtifactFile(tmp_path / "tx.csv"))


class TestBoundaryFeatures:
    """Validate authoritative Feature IDs and properties; geometry is validated by the shared boundary."""

    def test_exact_large_ids_annotations_and_semantic_property_id(self, tmp_path: Path) -> None:
        document = _boundary_document()
        features = document["features"]
        assert isinstance(features, list)
        features[0]["id"] = 2**64 - 1
        features[0]["properties"] = {"id": str(2**64 - 1), "objectType": "cell", "annotation": "retained"}
        validated, ids = boundary_features(document, ArtifactFile(tmp_path / "boundaries.geojson"))
        assert ids.tolist() == [2**64 - 1]
        assert ids.dtype == np.dtype(np.uint64)
        assert validated[0]["properties"] == {"id": str(2**64 - 1), "objectType": "cell", "annotation": "retained"}
        assert validated is features

    @pytest.mark.parametrize("patch", [{"type": "Feature"}, {"features": "not a list"}, {"features": []}])
    def test_invalid_collection(self, tmp_path: Path, patch: dict[str, object]) -> None:
        document = _boundary_document()
        document.update(patch)
        with pytest.raises(ReaderFormatError):
            boundary_features(document, ArtifactFile(tmp_path / "boundaries.geojson"))

    @pytest.mark.parametrize(
        "invalid",
        [
            "feature",
            "feature_type",
            "id_missing",
            "id_bool",
            "id_float",
            "id_zero",
            "property_type",
            "property_id_type",
            "property_conflict",
            "property_geometry",
        ],
    )
    def test_invalid_feature_identity_or_properties(self, tmp_path: Path, invalid: str) -> None:
        document = _boundary_document()
        features = document["features"]
        assert isinstance(features, list)
        feature = features[0]
        if invalid == "feature":
            features[0] = 1
        elif invalid == "feature_type":
            feature["type"] = "not a Feature"
        elif invalid == "id_missing":
            feature.pop("id")
            feature["properties"] = {"id": "1"}
        elif invalid == "id_bool":
            feature["id"] = True
        elif invalid == "id_float":
            feature["id"] = 1.0
        elif invalid == "id_zero":
            feature["id"] = 0
        elif invalid == "property_type":
            feature["properties"] = "not an object"
        elif invalid == "property_id_type":
            feature["properties"] = {"id": False}
        elif invalid == "property_geometry":
            feature["properties"] = {"geometry": None}
        else:
            feature["properties"] = {"id": "2"}
        with pytest.raises(ReaderFormatError):
            boundary_features(document, ArtifactFile(tmp_path / "boundaries.geojson"))

    def test_duplicate_semantic_ids(self, tmp_path: Path) -> None:
        document = _boundary_document()
        features = document["features"]
        assert isinstance(features, list)
        duplicate = features[0].copy()
        duplicate["id"] = 1
        features.append(duplicate)
        with pytest.raises(ReaderFormatError, match="duplicate boundary IDs"):
            boundary_features(document, ArtifactFile(tmp_path / "boundaries.geojson"))

    def test_multipart_null_properties_and_numeric_equivalent_id(self, tmp_path: Path) -> None:
        document = _boundary_document()
        features = document["features"]
        assert isinstance(features, list)
        feature = features[0]
        geometry = feature["geometry"]
        geometry["coordinates"] = [geometry["coordinates"]]
        geometry["type"] = "MultiPolygon"
        feature["properties"] = {"id": "01"}
        validated, ids = boundary_features(document, ArtifactFile(tmp_path / "boundaries.geojson"))
        assert ids.tolist() == [1]
        assert validated[0]["properties"] == {"id": "1"}
        feature["properties"] = None
        assert boundary_features(document, ArtifactFile(tmp_path / "boundaries.geojson"))[0][0]["properties"] is None


def _boundary_document() -> dict[str, object]:
    """Return independently owned raw JSON state for schema permutations."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "1",
                "properties": {"objectType": "cell"},
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            }
        ],
    }
