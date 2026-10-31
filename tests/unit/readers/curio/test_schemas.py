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
    DEFINING_FEATURES_KEY,
    STRING_DTYPE,
    normalize_moran,
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

    @pytest.mark.parametrize("columns", [[], ["gene", "gene"], ["gene", ""], ["gene", "null\x00field"]])
    def test_invalid_names(self, tmp_path: Path, columns: list[str]) -> None:
        """Reject ambiguous annotation headers before parser name repair."""
        with pytest.raises(ReaderFormatError, match="header"):
            validate_header(columns, ArtifactFile(tmp_path / "enrichment.tsv"))

    def test_moran_row_name_header_is_positional(self, tmp_path: Path) -> None:
        """Accept the empty first field emitted for R row names."""
        validate_header(["", "arbitrary_statistic"], ArtifactFile(tmp_path / "moran.tsv"), feature_ids=True)


class TestValidateText:
    """Validate lexical fields without interpreting their spellings."""

    def test_preserves_lexical_values_and_missingness(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"note": ["0007", "TRUE", "1e999", "NA", "", None]}, dtype=STRING_DTYPE)
        expected = frame.copy()

        assert validate_text(frame, ArtifactFile(tmp_path / "features.tsv")) is None

        pd.testing.assert_frame_equal(frame, expected)

    def test_nul_identifier_is_rejected(self, tmp_path: Path) -> None:
        """Reject embedded NUL before it can change identifier meaning."""
        with pytest.raises(ReaderFormatError, match="NUL"):
            validate_text(
                pd.DataFrame({"gene": ["gene\x00suffix"]}, dtype=STRING_DTYPE),
                ArtifactFile(tmp_path / "features.tsv"),
            )


class TestNormalizeMoran:
    """Parse only required statistics, preserving their exact numerical meaning."""

    def test_explicit_numeric_dtypes(self, tmp_path: Path) -> None:
        """Keep integers exact and parse scientific floating/boolean statistics."""
        frame = pd.DataFrame(
            {
                "integer": [str(2**53 + 1), "2"],
                "unsigned": [str(2**64 - 1), "2"],
                "float": ["0.10000000000000002", "-2.5e-1"],
                "boolean": ["TRUE", "FALSE"],
            },
            dtype=STRING_DTYPE,
        )

        assert normalize_moran(frame, ArtifactFile(tmp_path / "moran.tsv")) is None

        assert frame["integer"].tolist() == [2**53 + 1, 2]
        assert frame["unsigned"].tolist() == [2**64 - 1, 2]
        assert frame["float"].tolist() == [float("0.10000000000000002"), -0.25]
        assert frame["boolean"].tolist() == [True, False]
        assert frame.dtypes.tolist() == [
            pd.ArrowDtype(pa.int64()),
            pd.ArrowDtype(pa.uint64()),
            pd.ArrowDtype(pa.float64()),
            pd.ArrowDtype(pa.bool_()),
        ]

    @pytest.mark.parametrize("token", ["9007199254740993.0", "1e-1000"])
    def test_numeric_precision_loss(self, tmp_path: Path, token: str) -> None:
        """Reject decimal rounding and underflow at the annotation boundary."""
        with pytest.raises(ReaderFormatError, match="precision"):
            normalize_moran(pd.DataFrame({"stat": [token]}, dtype=STRING_DTYPE), ArtifactFile(tmp_path / "moran.tsv"))

    @pytest.mark.parametrize(
        "token", ["NA", "NaN", "text", "", "inf", "-inf", "1e999", "18446744073709551617", " 0.25 "]
    )
    def test_invalid_statistic(self, tmp_path: Path, token: str) -> None:
        with pytest.raises(ReaderFormatError, match=r"Moran statistic|precision"):
            normalize_moran(pd.DataFrame({"I": [token]}, dtype=STRING_DTYPE), ArtifactFile(tmp_path / "moran.tsv"))

    def test_boolean_and_numeric_tokens_cannot_mix(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="nonnumeric"):
            normalize_moran(
                pd.DataFrame({"I": ["TRUE", "1"]}, dtype=STRING_DTYPE), ArtifactFile(tmp_path / "moran.tsv")
            )

    def test_zero_float_statistics(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"I": ["0.0", "-0.0"]}, dtype=STRING_DTYPE)

        normalize_moran(frame, ArtifactFile(tmp_path / "moran.tsv"))

        np.testing.assert_array_equal(frame["I"], [0.0, 0.0])
        assert frame["I"].dtype == pd.ArrowDtype(pa.float64())


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
    """Headerless cluster rows have exactly two nonmissing positional fields."""

    def test_null_cluster(self, tmp_path: Path) -> None:
        """Require a nonmissing cluster assignment for each observation."""
        with pytest.raises(ReaderFormatError, match="cluster"):
            validate_clusters(
                pd.DataFrame({0: ["bead"], 1: [None]}, dtype=STRING_DTYPE),
                pd.Index(["bead"]),
                ArtifactFile(tmp_path / "clusters.tsv"),
            )


class TestValidateMetrics:
    """Metric categories cannot replace preserved H5AD uns or generated metadata."""

    @pytest.mark.parametrize("category", ["existing", DEFINING_FEATURES_KEY])
    def test_category_collision(self, tmp_path: Path, category: str) -> None:
        """Prevent metric categories from overwriting imported or generated uns annotations."""
        frame = pd.DataFrame({"arbitrary": ["Reads"], "result": ["1"], "Category": [category]}, dtype=STRING_DTYPE)

        with pytest.raises(ReaderFormatError, match="conflicting"):
            validate_metrics(frame, ArtifactFile(tmp_path / "metrics.csv"), {"existing"})


class TestValidateMoran:
    """Supplied dynamic statistics require known IDs and finite nonmissing values."""

    def test_missing_feature_id(self, tmp_path: Path) -> None:
        """Reject null feature identifiers before alignment."""
        frame = pd.DataFrame({"I": [1]}, index=pd.Index([None], dtype=STRING_DTYPE))

        with pytest.raises(ReaderFormatError, match="feature IDs"):
            validate_moran(frame, pd.DataFrame(index=["gene"]), ArtifactFile(tmp_path / "moran.tsv"))
