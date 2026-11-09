"""seqFISH field contracts, exact IDs, finite coordinates, and lazy point partitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.seqfish._schemas import (
    STRING_DTYPE,
    boundary_features,
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
        with pytest.raises(ReaderFormatError) as caught:
            validate_count_columns(columns, ArtifactFile(tmp_path / "counts.csv"))
        assert caught.value.context.artifact == "CellxGene"


class TestValidateCellColumns:
    """Project named metadata independent of input column order."""

    def test_order_and_extras(self, tmp_path: Path) -> None:
        assert validate_cell_columns(
            ["extra", "center_y", "area", "label", "center_x"], ArtifactFile(tmp_path / "cells.csv")
        ) == ("label", "area", "center_x", "center_y")

    def test_missing_fields(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="cell fields") as caught:
            validate_cell_columns(["label", "area"], ArtifactFile(tmp_path / "cells.csv"))
        assert caught.value.context.artifact == "CellCoordinates"


class TestValidateTranscriptColumns:
    """Declare optional plane and assignment fields without accepting repaired names."""

    def test_projection(self, tmp_path: Path) -> None:
        assert validate_transcript_columns(
            ["extra", "cell", "name", "y", "z", "x"], ArtifactFile(tmp_path / "tx.csv")
        ) == ("name", "x", "y", "z", "cell")

    def test_required(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="transcript fields") as caught:
            validate_transcript_columns(["x", "y"], ArtifactFile(tmp_path / "tx.csv"))
        assert caught.value.context.artifact == "TranscriptList"


class TestNormalizeTranscriptPartition:
    """Normalize only the executed partition and preserve unassigned events."""

    @pytest.mark.parametrize("assignments", [[], ["0"], ["2"], [str(2**53 + 1)], [str(2**63 + 1)], [str(2**64 - 1)]])
    def test_assignment_storage_is_partition_independent(self, tmp_path: Path, assignments: list[str]) -> None:
        frame = pd.DataFrame(
            {
                "name": ["A"] * len(assignments),
                "x": ["1"] * len(assignments),
                "y": ["2"] * len(assignments),
                "cell": assignments,
            },
            dtype=STRING_DTYPE,
        )
        result = normalize_transcript_partition(frame, features=("A",), path=ArtifactFile(tmp_path / "tx.csv"))
        assert result["cell"].dtype == pd.ArrowDtype(pa.uint64())
        assert result["cell"].tolist() == [int(value) for value in assignments]

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
        empty = pd.DataFrame({name: pd.Series(dtype=STRING_DTYPE) for name in ("name", "x", "y", "cell")})
        assert normalize_transcript_partition(empty, features=("A", "B"), path=path).empty

    @pytest.mark.parametrize("name", ["unknown", None])
    def test_unknown_features(self, tmp_path: Path, name: str | None) -> None:
        with pytest.raises(ReaderFormatError, match="vocabulary"):
            normalize_transcript_partition(
                pd.DataFrame({"name": [name]}), features=("A",), path=ArtifactFile(tmp_path / "tx.csv")
            )

    def test_plane_overflow(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"name": ["A"], "x": ["1"], "y": ["1"], "z": [str(2**63)]})
        with pytest.raises(ReaderFormatError, match="plane overflow") as caught:
            normalize_transcript_partition(frame, features=("A",), path=ArtifactFile(tmp_path / "tx.csv"))
        assert caught.value.context.artifact == "TranscriptList"


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
        elif invalid == "property_geometry":
            feature["properties"] = {"geometry": None}
        with pytest.raises(ReaderFormatError):
            boundary_features(document, ArtifactFile(tmp_path / "boundaries.geojson"))

    def test_duplicate_semantic_ids(self, tmp_path: Path) -> None:
        document = _boundary_document()
        features = document["features"]
        assert isinstance(features, list)
        duplicate = features[0].copy()
        duplicate["id"] = 1
        features.append(duplicate)
        with pytest.raises(ReaderFormatError, match="duplicate boundary IDs") as caught:
            boundary_features(document, ArtifactFile(tmp_path / "boundaries.geojson"))
        assert caught.value.context.artifact == "Boundaries"

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
        assert validated[0]["properties"] == {"id": "01"}
        feature["properties"] = None
        assert boundary_features(document, ArtifactFile(tmp_path / "boundaries.geojson"))[0][0]["properties"] is None

    @pytest.mark.parametrize("annotation", ["unrelated", False, None, 2, {"source": "other"}])
    def test_property_id_is_an_independent_annotation(self, tmp_path: Path, annotation: object) -> None:
        document = _boundary_document()
        features = document["features"]
        assert isinstance(features, list)
        features[0]["properties"] = {"id": annotation}
        validated, ids = boundary_features(document, ArtifactFile(tmp_path / "boundaries.geojson"))
        assert ids.tolist() == [1]
        assert validated[0]["properties"] == {"id": annotation}


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
