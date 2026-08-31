"""Integration tests for table linkage through real SpatialData models."""

from __future__ import annotations

from typing import cast

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import Point
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table

_CONTEXT = ReaderErrorContext(reader="integration", artifact="linked table")


def _adata(
    regions: list[str],
    instances: np.ndarray,
    *,
    expression: object | None = None,
    observation_only: bool = False,
) -> ad.AnnData:
    obs = pd.DataFrame({"region": regions, "instance": instances}, index=[f"o{i}" for i in range(len(regions))])
    if observation_only:
        return ad.AnnData(obs=obs, shape=(len(obs), 0))
    if expression is None:
        expression = sparse.csr_matrix(np.eye(len(obs), dtype=np.int16))
    return ad.AnnData(expression, obs=obs)


class TestParseLinkedTable:
    """Compose linkage with the installed public TableModel parser."""

    def test_single_region_normalizes_expression_and_layer(self) -> None:
        adata = _adata(
            ["cells", "cells"],
            np.asarray([1, 2], dtype=np.int64),
            expression=np.asarray([[1, 0], [0, 2]], dtype=np.int16),
        )
        adata.layers["counts"] = sparse.csc_matrix([[3, 0], [0, 4]], dtype=np.int32)

        with pytest.warns(UserWarning, match="categorical"):
            result = parse_linked_table(
                adata,
                TableLinkage(("cells",), "region", "instance"),
                context=_CONTEXT,
                expression_layers=("counts",),
                expected_instance_ids={"cells": pd.Index([2, 1])},
            )

        assert result is adata
        expression = result.X
        counts = result.layers["counts"]
        assert isinstance(expression, sparse.csr_array)
        assert expression.has_canonical_format
        assert isinstance(expression, sparse.sparray)
        assert not isinstance(expression, sparse.spmatrix)
        assert isinstance(counts, sparse.csr_array)
        assert counts.has_canonical_format
        assert not isinstance(counts, sparse.spmatrix)
        assert isinstance(result.obs["region"].dtype, pd.CategoricalDtype)
        assert result.uns[TableModel.ATTRS_KEY] == {
            "region": "cells",
            "region_key": "region",
            "instance_key": "instance",
        }

    def test_ordered_multi_region_allows_region_local_instances(self) -> None:
        adata = _adata(["b", "a"], np.asarray([7, 7], dtype=np.uint32))

        with pytest.warns(UserWarning, match="categorical"):
            result = parse_linked_table(
                adata,
                TableLinkage(("a", "b"), "region", "instance"),
                context=_CONTEXT,
                expected_instance_ids={
                    "b": pd.Index(np.asarray([7], dtype=np.uint64)),
                    "a": pd.CategoricalIndex(np.asarray([7], dtype=np.int16)),
                },
            )

        assert result.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == ["a", "b"]
        assert result.obs["region"].cat.categories.tolist() == ["a", "b"]

    def test_preserves_existing_categorical_region_storage(self) -> None:
        adata = _adata(["cells", "cells"], np.asarray([1, 2], dtype=np.int32))
        obs = cast("pd.DataFrame", adata.obs)
        obs["region"] = pd.Categorical(obs["region"])

        result = parse_linked_table(
            adata,
            TableLinkage(("cells",), "region", "instance"),
            context=_CONTEXT,
        )

        assert result is adata
        assert isinstance(result.obs["region"].dtype, pd.CategoricalDtype)

    def test_accepts_nonempty_observation_only_linked_table(self) -> None:
        adata = _adata(
            ["cells", "cells"],
            np.asarray([1, 2], dtype=np.int64),
            observation_only=True,
        )

        with pytest.warns(UserWarning, match="categorical"):
            result = parse_linked_table(
                adata,
                TableLinkage(("cells",), "region", "instance"),
                context=_CONTEXT,
            )

        assert result.X is None
        assert result.shape == (2, 0)

    @pytest.mark.parametrize(
        ("regions", "instances", "declared", "match"),
        [
            ([], np.asarray([], dtype=np.int64), ("cells",), "no observed regions"),
            (["cells"], np.asarray([1], dtype=np.int64), ("cells", "unused"), "observed regions"),
            (["cells", "cells"], np.asarray([1, 1], dtype=np.int64), ("cells",), "duplicate"),
        ],
    )
    def test_rejects_invalid_region_and_instance_relationships(
        self,
        regions: list[str],
        instances: np.ndarray,
        declared: tuple[str, ...],
        match: str,
    ) -> None:
        adata = _adata(regions, instances, observation_only=True)

        with pytest.raises(ReaderFormatError, match=match):
            parse_linked_table(
                adata,
                TableLinkage(declared, "region", "instance"),
                context=_CONTEXT,
            )

    def test_composes_with_shapes_and_spatialdata(self) -> None:
        shape_index = pd.Index(np.asarray([10, 20], dtype=np.int64), name="instance")
        frame = gpd.GeoDataFrame(
            {ShapesModel.RADIUS_KEY: [1.0, 1.0]},
            geometry=[Point(0, 0), Point(1, 1)],
            index=shape_index,
        )
        shapes = ShapesModel.parse(frame)
        adata = _adata(["cells", "cells"], np.asarray([20, 10], dtype=np.int64))

        with pytest.warns(UserWarning, match="categorical"):
            table = parse_linked_table(
                adata,
                TableLinkage(("cells",), "region", "instance"),
                context=_CONTEXT,
                expected_instance_ids={"cells": shapes.index},
            )
        sdata = SpatialData(shapes={"cells": shapes}, tables={"table": table})

        assert sdata.tables["table"] is table
        assert isinstance(sdata.tables["table"].X, sparse.csr_array)
        assert sdata.shapes["cells"].index.tolist() == [10, 20]
