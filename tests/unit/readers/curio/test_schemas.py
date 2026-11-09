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
    normalize_moran,
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

    @pytest.mark.parametrize("token", ["9007199254740993.0", "1e-1000", " 0.25 ", "\t-2.5e-1\t"])
    def test_native_float_rounding(self, tmp_path: Path, token: str) -> None:
        frame = pd.DataFrame({"stat": [token]}, dtype=STRING_DTYPE)

        normalize_moran(frame, ArtifactFile(tmp_path / "moran.tsv"))

        assert frame["stat"].tolist() == [float(token)]
        assert frame["stat"].dtype == pd.ArrowDtype(pa.float64())

    @pytest.mark.parametrize("token", ["NA", "NaN", "text", "", "inf", "-inf", "1e999", "18446744073709551617"])
    def test_invalid_statistic(self, tmp_path: Path, token: str) -> None:
        with pytest.raises(ReaderFormatError, match="Moran statistic"):
            normalize_moran(pd.DataFrame({"I": [token]}, dtype=STRING_DTYPE), ArtifactFile(tmp_path / "moran.tsv"))

    def test_boolean_and_numeric_tokens_cannot_mix(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match="invalid Moran statistic"):
            normalize_moran(
                pd.DataFrame({"I": ["TRUE", "1"]}, dtype=STRING_DTYPE), ArtifactFile(tmp_path / "moran.tsv")
            )

    def test_zero_float_statistics(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"I": ["0.0", "-0.0"]}, dtype=STRING_DTYPE)

        normalize_moran(frame, ArtifactFile(tmp_path / "moran.tsv"))

        np.testing.assert_array_equal(frame["I"], [0.0, 0.0])
        assert frame["I"].dtype == pd.ArrowDtype(pa.float64())

    @pytest.mark.parametrize(
        ("tokens", "dtype", "expected"),
        [
            (["\t-9223372036854775808 ", "0x10", "0007", "-0"], pa.int64(), [-(2**63), 16, 7, 0]),
            (["0xffffffffffffffff", "0", "-0", "9223372036854775808"], pa.uint64(), [2**64 - 1, 0, 0, 2**63]),
            (["0X7FFFFFFFFFFFFFFF", "2", "0", "-1"], pa.int64(), [2**63 - 1, 2, 0, -1]),
            (["TRUE", "False", "true", "FALSE"], pa.bool_(), [True, False, True, False]),
            (["9007199254740993", "0.5", "1e-400", "-2"], pa.float64(), [float(2**53 + 1), 0.5, 0.0, -2.0]),
        ],
    )
    def test_multichunk_exact_domains_and_row_order(
        self, tmp_path: Path, tokens: list[str], dtype: pa.DataType, expected: list[int | float | bool]
    ) -> None:
        """A statistic's type and values are independent of Arrow CSV chunk boundaries."""
        chunks = pa.chunked_array([tokens[:1], tokens[1:3], tokens[3:]], type=pa.string())
        source = pd.Series(pd.arrays.ArrowExtensionArray(chunks), index=pd.Index([3, 1, 4, 2], name="row"))
        frame = pd.DataFrame({"statistic": source})
        frame.columns.name = "statistics"

        normalize_moran(frame, ArtifactFile(tmp_path / "moran.tsv"))

        assert frame["statistic"].tolist() == expected
        assert frame["statistic"].dtype == pd.ArrowDtype(dtype)
        assert frame.index.equals(source.index)
        assert frame.index.name == "row"
        assert frame.columns.name == "statistics"

    @pytest.mark.parametrize(
        "tokens",
        [
            ["18446744073709551616"],
            ["-9223372036854775809"],
            ["-1", "9223372036854775808"],
            ["+1"],
            ["+9007199254740993"],
            ["-0x1"],
            ["-0x0", "18446744073709551615"],
            ["0x10000000000000000"],
        ],
    )
    def test_integer_columns_never_fall_back_to_float(self, tmp_path: Path, tokens: list[str]) -> None:
        with pytest.raises(ReaderFormatError, match="Moran statistic"):
            normalize_moran(pd.DataFrame({"stat": tokens}, dtype=STRING_DTYPE), ArtifactFile(tmp_path / "moran.tsv"))

    def test_mixed_column_declares_float_domain(self, tmp_path: Path) -> None:
        """A fraction declares float64; integer-looking values then use native rounding."""
        frame = pd.DataFrame({"stat": ["18446744073709551616", "0.5"]}, dtype=STRING_DTYPE)

        normalize_moran(frame, ArtifactFile(tmp_path / "moran.tsv"))

        assert frame["stat"].tolist() == [float(2**64), 0.5]
        assert frame["stat"].dtype == pd.ArrowDtype(pa.float64())

    @pytest.mark.parametrize("token", [" TRUE", "False\t", "1_0"])
    def test_only_declared_boolean_and_numeric_spellings(self, tmp_path: Path, token: str) -> None:
        with pytest.raises(ReaderFormatError, match="invalid Moran statistic"):
            normalize_moran(pd.DataFrame({"stat": [token]}, dtype=STRING_DTYPE), ArtifactFile(tmp_path / "moran.tsv"))


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

    def test_missing_feature_id(self, tmp_path: Path) -> None:
        """Reject null feature identifiers before alignment."""
        frame = pd.DataFrame({"I": [1]}, index=pd.Index([None], dtype=STRING_DTYPE))

        with pytest.raises(ReaderFormatError, match="feature IDs"):
            validate_moran(frame, pd.DataFrame(index=["gene"]), ArtifactFile(tmp_path / "moran.tsv"))
