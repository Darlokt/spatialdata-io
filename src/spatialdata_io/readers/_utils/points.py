"""Shared contracts for lazy transcript points."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Generator


def categorize_features(values: pd.Series, vocabulary: pd.Index, /) -> pd.Categorical | None:
    """Return known categories in vocabulary order, or ``None`` for unknown or missing values.

    ``vocabulary`` contains unique source feature names. Point partitions can
    repeat those names millions of times, so factorize on the source dtype first
    and look up only the distinct values.
    """
    inverse, distinct = pd.factorize(values, use_na_sentinel=False)
    codes = vocabulary.get_indexer(distinct)
    if (codes < 0).any():
        return None
    return pd.Categorical.from_codes(codes[inverse], categories=vocabulary)


@contextmanager
def suppress_points_index_warning() -> Generator[None, None, None]:
    """Let PointsModel retain intentionally unknown divisions without warning about sorting."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The index of the dataframe is not monotonic increasing\. It is recommended to sort",
            category=UserWarning,
        )
        yield
