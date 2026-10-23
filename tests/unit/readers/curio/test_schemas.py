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
    normalize_metadata,
    validate_clusters,
    validate_counts,
    validate_header,
    validate_metrics,
    validate_moran,
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


class TestNormalizeMetadata:
    """Retain lexical identifiers and normalize dynamic primitive fields exactly."""

    def test_explicit_dtypes_and_missing_values(self, tmp_path: Path) -> None:
        """Normalize dynamic primitives while retaining exact integers and lexical IDs."""
        frame = pd.DataFrame(
            {
                "gene": ["001", "NA"],
                "integer": [str(2**53 + 1), "2"],
                "unsigned": [str(2**64 - 1), "2"],
                "float": ["0.10000000000000002", None],
                "boolean": ["TRUE", None],
                "extra": ["NULL", "retained"],
            },
            dtype=STRING_DTYPE,
        )

        normalized = normalize_metadata(frame, ArtifactFile(tmp_path / "features.tsv"), strings=("gene",))

        assert normalized is frame
        assert normalized["gene"].tolist() == ["001", "NA"]
        assert normalized["integer"].tolist() == [2**53 + 1, 2]
        assert normalized["unsigned"].tolist() == [2**64 - 1, 2]
        assert normalized["float"].iloc[0] == float("0.10000000000000002")
        assert normalized["float"].isna().tolist() == [False, True]
        assert normalized["boolean"].iloc[0]
        assert normalized["boolean"].isna().tolist() == [False, True]
        assert normalized["extra"].tolist() == ["NULL", "retained"]
        assert normalized.dtypes.tolist() == [
            STRING_DTYPE,
            pd.ArrowDtype(pa.int64()),
            pd.ArrowDtype(pa.uint64()),
            pd.ArrowDtype(pa.float64()),
            pd.ArrowDtype(pa.bool_()),
            STRING_DTYPE,
        ]

    @pytest.mark.parametrize("token", ["9007199254740993.0", "1e-1000"])
    def test_numeric_precision_loss(self, tmp_path: Path, token: str) -> None:
        """Reject decimal rounding and underflow at the annotation boundary."""
        with pytest.raises(ReaderFormatError, match="precision"):
            normalize_metadata(
                pd.DataFrame({"stat": [token]}, dtype=STRING_DTYPE), ArtifactFile(tmp_path / "moran.tsv")
            )

    def test_nonnumeric_and_overflow_lexemes_remain_text(self, tmp_path: Path) -> None:
        """Preserve text that cannot be represented as finite primitive numeric metadata."""
        frame = pd.DataFrame({"cluster": ["NaN", "1"], "extra": ["1e999", "2"]}, dtype=STRING_DTYPE)

        normalize_metadata(frame, ArtifactFile(tmp_path / "features.tsv"))

        assert frame["cluster"].tolist() == ["NaN", "1"]
        assert frame["extra"].tolist() == ["1e999", "2"]

    def test_nul_identifier_is_rejected(self, tmp_path: Path) -> None:
        """Reject embedded NUL before it can change identifier meaning."""
        with pytest.raises(ReaderFormatError, match="NUL"):
            normalize_metadata(
                pd.DataFrame({"gene": ["gene\x00suffix"]}, dtype=STRING_DTYPE),
                ArtifactFile(tmp_path / "features.tsv"),
                strings=("gene",),
            )


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
