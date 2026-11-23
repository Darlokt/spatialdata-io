"""Known, source-ordered categorical vocabularies for large point partitions."""

from __future__ import annotations

import warnings

import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.points import categorize_features, suppress_points_index_warning


class TestCategorizeFeatures:
    @pytest.mark.parametrize("values", [[], ["001", "é", "001", "z", "é"]])
    def test_matches_direct_lookup_and_keeps_unused_categories(self, values: list[str]) -> None:
        series = pd.Series(values, dtype=pd.ArrowDtype(pa.string()))
        features = pd.Index(["z", "unused", "é", "001"])
        expected = pd.Categorical.from_codes(features.get_indexer(pd.Index(series.array)), categories=features)
        actual = categorize_features(series, features)
        assert actual is not None
        pd.testing.assert_index_equal(actual.categories, features)
        assert actual.equals(expected)

    @pytest.mark.parametrize("bad", [None, "", "unknown"])
    def test_unknown_or_missing_values_fail(self, bad: str | None) -> None:
        series = pd.Series(["001", bad, "001"], dtype=pd.ArrowDtype(pa.string()))
        assert categorize_features(series, pd.Index(["001"])) is None


class TestSuppressPointsIndexWarning:
    def test_scope_and_unrelated_warnings(self) -> None:
        message = "The index of the dataframe is not monotonic increasing. It is recommended to sort the data"
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with suppress_points_index_warning():
                warnings.warn(message, UserWarning, stacklevel=1)
                warnings.warn("an unrelated point warning", UserWarning, stacklevel=1)
            warnings.warn(message, UserWarning, stacklevel=1)
        assert [str(item.message) for item in captured] == ["an unrelated point warning", message]
