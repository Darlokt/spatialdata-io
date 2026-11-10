"""Curio positional exceptions, extensible annotations, and exact Arrow schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.curio._schemas import (
    CLUSTER_TYPES,
    DEFINING_FEATURES_KEY,
    STRING_DTYPE,
    moran_types,
    text_types,
    validate_clusters,
    validate_counts,
    validate_header,
    validate_metrics,
    validate_moran,
    validate_text,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestValidateHeader:
    """Reject repaired or unnamed schema fields at the raw header boundary."""

    @pytest.mark.parametrize("columns", [[], ["gene", ""], ["gene", "null\x00field"]])
    def test_invalid_names(self, tmp_path: Path, columns: list[str]) -> None:
        """Reject ambiguous annotation headers before parser name repair."""
        with pytest.raises(ReaderFormatError, match="header"):
            validate_header(columns, "features", ArtifactFile(tmp_path / "enrichment.tsv"))

    def test_moran_row_name_header_is_positional(self, tmp_path: Path) -> None:
        """Accept the empty first field emitted for R row names."""
        validate_header(
            ["", "arbitrary_statistic"], "Moran features", ArtifactFile(tmp_path / "moran.tsv"), feature_ids=True
        )


class TestTextTypes:
    """The positional and dynamic text schemas have one source of names and types."""

    def test_cluster_fields(self) -> None:
        assert pa.schema([("bead_id", pa.string()), ("cluster", pa.string())]) == CLUSTER_TYPES

    def test_named_field_order_and_empty_row_name(self) -> None:
        assert text_types(["", "arbitrary", "001"]) == pa.schema(
            [("", pa.string()), ("arbitrary", pa.string()), ("001", pa.string())]
        )

    def test_moran_feature_ids_and_float_statistics(self) -> None:
        assert moran_types(["", "MoransI_observed", "rank"]) == pa.schema(
            [("", pa.string()), ("MoransI_observed", pa.float64()), ("rank", pa.float64())]
        )


class TestValidateText:
    """Validate lexical fields without interpreting their spellings."""

    def test_preserves_lexical_values_and_missingness(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"note": ["0007", "TRUE", "1e999", "NA", "", None]}, dtype=STRING_DTYPE)
        expected = frame.copy()

        assert validate_text(frame, "features", ArtifactFile(tmp_path / "features.tsv")) is None

        pd.testing.assert_frame_equal(frame, expected)

    def test_nul_identifier_is_rejected(self, tmp_path: Path) -> None:
        """Reject embedded NUL before it can change identifier meaning."""
        with pytest.raises(ReaderFormatError, match="NUL") as raised:
            validate_text(
                pd.DataFrame({"gene": ["gene\x00suffix"]}, dtype=STRING_DTYPE),
                "features",
                ArtifactFile(tmp_path / "features.tsv"),
            )

        assert raised.value.context.artifact == "features"


class TestValidateCounts:
    """Preserved annotations must not collide with reader-owned destinations."""

    @pytest.mark.parametrize("column", ["cluster", "region", "instance_id"])
    def test_reserved_observation_column(self, tmp_path: Path, column: str) -> None:
        """Prevent reader enrichment from overwriting preserved observation annotations."""
        table = ad.AnnData(np.ones((1, 1)), obs=pd.DataFrame({column: ["source"]}, index=["bead"]))

        with pytest.raises(ReaderFormatError, match="reserved"):
            validate_counts(table, ArtifactFile(tmp_path / "counts.h5ad"))

    def test_reserved_uns_frame(self, tmp_path: Path) -> None:
        """Prevent defining-feature metadata from replacing an imported uns value."""
        table = ad.AnnData(np.ones((1, 1)), uns={DEFINING_FEATURES_KEY: "source"})

        with pytest.raises(ReaderFormatError, match="reserved"):
            validate_counts(table, ArtifactFile(tmp_path / "counts.h5ad"))


class TestValidateClusters:
    """One nonempty cluster label per H5AD observation."""

    @pytest.mark.parametrize(("bead", "cluster"), [("", "1"), ("bead", "")])
    def test_empty_field(self, tmp_path: Path, bead: str, cluster: str) -> None:
        with pytest.raises(ReaderFormatError, match="empty bead ID or cluster label") as raised:
            validate_clusters(
                pd.DataFrame({"bead_id": [bead], "cluster": [cluster]}, dtype=STRING_DTYPE),
                pd.Index(["bead"]),
                ArtifactFile(tmp_path / "clusters.tsv"),
            )

        assert raised.value.context.artifact == "cluster assignments"

    @pytest.mark.parametrize("beads", [["bead", "bead"], ["other"], []])
    def test_bead_population_must_match(self, tmp_path: Path, beads: list[str]) -> None:
        with pytest.raises(ReaderFormatError, match="mismatched bead IDs"):
            validate_clusters(
                pd.DataFrame({"bead_id": beads, "cluster": ["1"] * len(beads)}, dtype=STRING_DTYPE),
                pd.Index(["bead"]),
                ArtifactFile(tmp_path / "clusters.tsv"),
            )


class TestValidateMetrics:
    """Metric categories cannot replace preserved H5AD uns or generated metadata."""

    @pytest.mark.parametrize(
        "category", ["existing", DEFINING_FEATURES_KEY, "spatialdata_io_positions", "spatialdata_attrs"]
    )
    def test_category_collision(self, tmp_path: Path, category: str) -> None:
        """Prevent metric categories from overwriting imported or generated uns annotations."""
        frame = pd.DataFrame({"arbitrary": ["Reads"], "result": ["1"], "Category": [category]}, dtype=STRING_DTYPE)

        with pytest.raises(ReaderFormatError, match="conflicting"):
            validate_metrics(frame, ArtifactFile(tmp_path / "metrics.csv"), {"existing"})


class TestValidateMoran:
    """Supplied dynamic statistics require known IDs and finite nonmissing values."""

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_statistic(self, tmp_path: Path, value: float) -> None:
        """Reject the nan and inf spellings that Arrow parses as float64 values."""
        # pa.array keeps NaN a value, as Arrow's CSV parser does, rather than a null.
        frame = pd.DataFrame({"I": pd.arrays.ArrowExtensionArray(pa.array([0.5, value]))}, index=["a", "b"])

        with pytest.raises(ReaderFormatError, match="nonfinite Moran statistics"):
            validate_moran(frame, pd.DataFrame(index=["a", "b"]), ArtifactFile(tmp_path / "moran.tsv"))

    def test_missing_feature_id(self, tmp_path: Path) -> None:
        """Reject null feature identifiers before alignment."""
        frame = pd.DataFrame({"I": [1]}, index=pd.Index([None], dtype=STRING_DTYPE))

        with pytest.raises(ReaderFormatError, match="feature IDs"):
            validate_moran(frame, pd.DataFrame(index=["gene"]), ArtifactFile(tmp_path / "moran.tsv"))
