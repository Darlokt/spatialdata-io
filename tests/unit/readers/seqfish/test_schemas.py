"""seqFISH Arrow types, field contracts, finite coordinates, and lazy point partitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.seqfish import _schemas as seqfish_schemas
from spatialdata_io.readers.seqfish._schemas import (
    boundary_features,
    count_types,
    normalize_transcript_partition,
    transcript_types,
    validate_cell_columns,
    validate_cell_ids,
)

if TYPE_CHECKING:
    from pathlib import Path

_FEATURES = pd.Index(["A", "B", "ZeroCount"])


def _partition(**columns: pa.Array) -> pd.DataFrame:
    """Return a typed partition as the lazy CSV blocks produce it."""
    return pa.table(columns).to_pandas(types_mapper=pd.ArrowDtype)


class TestCountTypes:
    """Type the positional identity and every feature in exact spelling and order."""

    @pytest.mark.parametrize("first", ["", "label"])
    def test_positional_identity_then_counts(self, tmp_path: Path, first: str) -> None:
        types = count_types((first, "Gene-B", "GeneA"), ArtifactFile(tmp_path / "counts.csv"))
        assert types == pa.schema([(first, pa.uint64()), ("Gene-B", pa.uint64()), ("GeneA", pa.uint64())])

    @pytest.mark.parametrize("header", [(), ("",), ("", ""), ("", "A\x00"), ("x", "")])
    def test_unusable_headers(self, tmp_path: Path, header: tuple[str, ...]) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            count_types(header, ArtifactFile(tmp_path / "counts.csv"))
        assert caught.value.context.artifact == "CellxGene"


class TestValidateCellColumns:
    """Require named metadata independent of input column order."""

    @pytest.mark.parametrize(
        "header", [("extra", "center_y", "area", "label", "center_x"), ("", "label", "area", "center_x", "center_y")]
    )
    def test_order_extras_and_unnamed_index(self, tmp_path: Path, header: tuple[str, ...]) -> None:
        validate_cell_columns(header, ArtifactFile(tmp_path / "cells.csv"))

    @pytest.mark.parametrize("header", [("label", "area"), ("label", "area", "center_x")])
    def test_missing_fields(self, tmp_path: Path, header: tuple[str, ...]) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            validate_cell_columns(header, ArtifactFile(tmp_path / "cells.csv"))
        assert caught.value.context.artifact == "CellCoordinates"


class TestTranscriptTypes:
    """Declare optional plane and assignment fields; ignore other annotations."""

    def test_projection_and_types(self, tmp_path: Path) -> None:
        types = transcript_types(("extra", "cell", "name", "y", "z", "x"), ArtifactFile(tmp_path / "tx.csv"))
        assert types == {
            "name": pa.string(),
            "x": pa.float64(),
            "y": pa.float64(),
            "z": pa.uint64(),
            "cell": pa.uint64(),
        }
        # Ignored annotations may have any name, such as the unnamed index pandas writes.
        header = ("", "y", "x", "name", "\x00")
        assert list(transcript_types(header, ArtifactFile(tmp_path / "tx.csv"))) == ["name", "x", "y"]

    def test_required_fields(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError) as caught:
            transcript_types(("x", "y"), ArtifactFile(tmp_path / "tx.csv"))
        assert caught.value.context.artifact == "TranscriptList"


class TestNormalizeTranscriptPartition:
    """Normalize only the executed partition and preserve unassigned events."""

    def test_partition_values_and_types(self, tmp_path: Path) -> None:
        frame = _partition(
            name=pa.array(["B", "A"]),
            x=pa.array([0.1, 2.0]),
            y=pa.array([3.0, 4.0]),
            z=pa.array([0, 2**64 - 1], pa.uint64()),
            cell=pa.array([0, 2**64 - 1], pa.uint64()),
        )
        result = normalize_transcript_partition(frame, features=_FEATURES, path=ArtifactFile(tmp_path / "tx.csv"))
        assert result.columns.tolist() == ["name", "x", "y", "cell", "z_raw"]
        assert result["name"].cat.categories.tolist() == ["A", "B", "ZeroCount"]
        assert result["name"].tolist() == ["B", "A"]
        assert result["x"].dtype == np.dtype(np.float64)
        assert result["y"].dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(result["x"].to_numpy(), [0.1, 2.0])
        assert result["cell"].dtype == pd.ArrowDtype(pa.uint64())
        assert result["cell"].tolist() == [0, 2**64 - 1]
        assert result["z_raw"].dtype == pd.ArrowDtype(pa.uint64())
        assert result["z_raw"].tolist() == [0, 2**64 - 1]

    def test_empty_partition(self, tmp_path: Path) -> None:
        frame = _partition(name=pa.array([], pa.string()), x=pa.array([], pa.float64()), y=pa.array([], pa.float64()))
        result = normalize_transcript_partition(frame, features=_FEATURES, path=ArtifactFile(tmp_path / "tx.csv"))
        assert result.empty
        assert result["name"].cat.categories.tolist() == _FEATURES.tolist()

    @pytest.mark.parametrize(
        ("columns", "found"),
        [
            ({"name": ["unknown"], "x": [1.0], "y": [1.0]}, "unknown transcript names"),
            ({"name": ["A"], "x": [float("nan")], "y": [1.0]}, "nonfinite 'x' values"),
            ({"name": ["A"], "x": [1.0], "y": [float("-inf")]}, "nonfinite 'y' values"),
        ],
    )
    def test_invalid_values(self, tmp_path: Path, columns: dict[str, list[object]], found: str) -> None:
        frame = _partition(**{name: pa.array(values) for name, values in columns.items()})
        with pytest.raises(ReaderFormatError) as caught:
            normalize_transcript_partition(frame, features=_FEATURES, path=ArtifactFile(tmp_path / "tx.csv"))
        assert caught.value.found == found
        assert caught.value.context.artifact == "TranscriptList"


class TestValidateCellIds:
    """Keep typed source identifiers exact through positive-domain and uniqueness checks."""

    @pytest.mark.parametrize("chunked", [False, True])
    def test_full_uint64_order(self, tmp_path: Path, *, chunked: bool) -> None:
        source = pa.array([2**64 - 1, 1, 2**63, 2**64 - 2], pa.uint64())
        ids = pa.chunked_array([source.slice(0, 2), source.slice(2)]) if chunked else source
        index = validate_cell_ids(
            ids, context=seqfish_schemas.COUNTS_CONTEXT, path=ArtifactFile(tmp_path / "counts.csv")
        )
        assert index.tolist() == [2**64 - 1, 1, 2**63, 2**64 - 2]
        assert index.dtype == pd.ArrowDtype(pa.uint64())

    def test_empty(self, tmp_path: Path) -> None:
        index = validate_cell_ids(
            pa.array([], pa.uint64()), context=seqfish_schemas.CELLS_CONTEXT, path=ArtifactFile(tmp_path / "cells.csv")
        )
        assert index.empty
        assert index.dtype == pd.ArrowDtype(pa.uint64())

    @pytest.mark.parametrize(
        ("values", "found"),
        [([1, 0], "nonpositive cell IDs"), ([2**64 - 1, 2**64 - 1], "duplicate cell IDs")],
    )
    def test_invalid_population(self, tmp_path: Path, values: list[int], found: str) -> None:
        with pytest.raises(ReaderFormatError, match=found) as caught:
            validate_cell_ids(
                pa.array(values, pa.uint64()),
                context=seqfish_schemas.CELLS_CONTEXT,
                path=ArtifactFile(tmp_path / "cells.csv"),
            )
        assert caught.value.context.artifact == "CellCoordinates"


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
        assert ids.dtype == pd.ArrowDtype(pa.uint64())
        assert validated[0]["properties"] == {"id": str(2**64 - 1), "objectType": "cell", "annotation": "retained"}
        assert validated is features

    @pytest.mark.parametrize(("identifier", "expected"), [("01", 1), (7, 7), (str(2**64 - 1), 2**64 - 1)])
    def test_ids_are_cast_with_arrow(self, tmp_path: Path, identifier: str | int, expected: int) -> None:
        document = _boundary_document()
        features = document["features"]
        assert isinstance(features, list)
        features[0]["id"] = identifier
        assert boundary_features(document, ArtifactFile(tmp_path / "boundaries.geojson"))[1].tolist() == [expected]

    @pytest.mark.parametrize("identifier", ["1.0", -1, 2**64], ids=["fraction", "negative", "overflow"])
    def test_ids_that_are_not_uint64(self, tmp_path: Path, identifier: str | int) -> None:
        document = _boundary_document()
        features = document["features"]
        assert isinstance(features, list)
        features[0]["id"] = identifier
        with pytest.raises(ReaderFormatError, match="feature ids that are not integers") as caught:
            boundary_features(document, ArtifactFile(tmp_path / "boundaries.geojson"))
        assert caught.value.context.artifact == "Boundaries"

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

    @pytest.mark.parametrize("identifier", [1, 2**64 - 1])
    def test_duplicate_semantic_ids(self, tmp_path: Path, identifier: int) -> None:
        document = _boundary_document()
        features = document["features"]
        assert isinstance(features, list)
        features[0]["id"] = str(identifier)
        duplicate = features[0].copy()
        duplicate["id"] = identifier
        features.append(duplicate)
        with pytest.raises(ReaderFormatError, match="duplicate cell IDs") as caught:
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
