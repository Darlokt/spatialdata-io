"""Tests for validated table linkage through the public SpatialData parser."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import PropertyMock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from spatialdata.models import TableModel

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import TableLinkage, parse_linked_table

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


_CONTEXT = ReaderErrorContext(reader="test_reader", artifact="cell table", detected_version="1")


def _table(
    *,
    regions: pd.Series | list[object] | np.ndarray | None = None,
    instances: pd.Series | pd.Categorical | np.ndarray | list[object] | None = None,
    x: object | None = None,
) -> ad.AnnData:
    if regions is None:
        regions = ["cells", "cells"]
    if instances is None:
        instances = np.asarray([1, 2], dtype=np.int64)
    index = pd.Index(["o1", "o2"][: len(regions)])
    region_values = regions.set_axis(index) if isinstance(regions, pd.Series) else regions
    instance_values = instances.set_axis(index) if isinstance(instances, pd.Series) else instances
    obs = pd.DataFrame({"region": region_values, "instance": instance_values}, index=index)
    expression = sparse.csr_array(np.eye(len(obs), dtype=np.int16)) if x is None else x
    return ad.AnnData(expression, obs=obs)


def _public_parser_stub(
    adata: ad.AnnData,
    region: str | list[str] | None = None,
    region_key: str | None = None,
    instance_key: str | None = None,
    *,
    overwrite_metadata: bool = False,
) -> ad.AnnData:
    assert region is not None
    assert region_key is not None
    assert instance_key is not None
    assert not overwrite_metadata
    adata.uns[TableModel.ATTRS_KEY] = {
        TableModel.REGION_KEY: region,
        TableModel.REGION_KEY_KEY: region_key,
        TableModel.INSTANCE_KEY: instance_key,
    }
    obs = cast("pd.DataFrame", adata.obs)
    obs[region_key] = pd.Categorical(obs[region_key])
    return adata


class TestTableLinkage:
    """Validate the immutable reader-authored linkage descriptor."""

    @pytest.mark.parametrize(
        ("regions", "region_key", "instance_key", "match"),
        [
            ((), "region", "instance", "at least one"),
            (("",), "region", "instance", "empty names"),
            (("a", "b", "a"), "region", "instance", "duplicates"),
            (("a",), "", "instance", "nonempty"),
            (("a",), "region", "", "nonempty"),
            (("a",), "key", "key", "distinct"),
        ],
    )
    def test_rejects_invalid_relational_state(
        self,
        regions: tuple[str, ...],
        region_key: str,
        instance_key: str,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            TableLinkage(regions, region_key, instance_key)

    def test_is_frozen_and_slotted(self) -> None:
        linkage = TableLinkage(("cells",), "region", "instance")

        assert not hasattr(linkage, "__dict__")
        assert vars(TableLinkage)["__dataclass_params__"].frozen


class TestParseLinkedTable:
    """Validate linkage, normalization, parser isolation, and ownership."""

    def test_calls_normalization_and_parser_once_in_order(self, mocker: MockerFixture) -> None:
        adata = _table(x=np.asarray([[1, 0], [0, 2]], dtype=np.int16))
        events: list[str] = []
        real_normalize = __import__(
            "spatialdata_io.readers._utils.table._linkage",
            fromlist=["normalize_owned_anndata"],
        ).normalize_owned_anndata

        def normalize(table: ad.AnnData, *, expression_layers: tuple[str, ...] = ()) -> ad.AnnData:
            events.append("normalize")
            return real_normalize(table, expression_layers=expression_layers)

        def parse(
            table: ad.AnnData,
            region: str | list[str] | None = None,
            region_key: str | None = None,
            instance_key: str | None = None,
            *,
            overwrite_metadata: bool = False,
        ) -> ad.AnnData:
            events.append("parse")
            return _public_parser_stub(
                table,
                region=region,
                region_key=region_key,
                instance_key=instance_key,
                overwrite_metadata=overwrite_metadata,
            )

        normalization = mocker.patch(
            "spatialdata_io.readers._utils.table._linkage.normalize_owned_anndata",
            side_effect=normalize,
        )
        parser = mocker.patch.object(TableModel, "parse", side_effect=parse)

        result = parse_linked_table(
            adata,
            TableLinkage(("cells",), "region", "instance"),
            context=_CONTEXT,
        )

        assert result is adata
        assert events == ["normalize", "parse"]
        normalization.assert_called_once_with(adata, expression_layers=())
        parser.assert_called_once_with(adata, region="cells", region_key="region", instance_key="instance")

    def test_passes_ordered_region_list(self, mocker: MockerFixture) -> None:
        adata = _table(regions=["b", "a"], instances=np.asarray([1, 1], dtype=np.int32))
        parser = mocker.patch.object(TableModel, "parse", side_effect=_public_parser_stub)

        parse_linked_table(
            adata,
            TableLinkage(("a", "b"), "region", "instance"),
            context=_CONTEXT,
        )

        parser.assert_called_once_with(adata, region=["a", "b"], region_key="region", instance_key="instance")

    def test_rejects_view_before_linkage_validation(self, mocker: MockerFixture) -> None:
        view = _table()[:1]
        validation = mocker.patch("spatialdata_io.readers._utils.table._linkage._validated_pairs")
        parser = mocker.patch.object(TableModel, "parse")

        with pytest.raises(ValueError, match="not a view"):
            parse_linked_table(
                view,
                TableLinkage(("cells",), "missing", "also_missing"),
                context=_CONTEXT,
            )

        validation.assert_not_called()
        parser.assert_not_called()

    def test_rejects_backed_table_before_linkage_validation(self, mocker: MockerFixture) -> None:
        adata = _table()
        mocker.patch.object(ad.AnnData, "isbacked", new_callable=PropertyMock, return_value=True)
        validation = mocker.patch("spatialdata_io.readers._utils.table._linkage._validated_pairs")
        parser = mocker.patch.object(TableModel, "parse")

        with pytest.raises(TypeError, match="not a backed table"):
            parse_linked_table(
                adata,
                TableLinkage(("cells",), "missing", "also_missing"),
                context=_CONTEXT,
            )

        validation.assert_not_called()
        parser.assert_not_called()

    def test_rejects_stale_metadata_without_mutation(self, mocker: MockerFixture) -> None:
        adata = _table()
        stale = {"region": "old"}
        adata.uns[TableModel.ATTRS_KEY] = stale
        expression = adata.X
        parser = mocker.patch.object(TableModel, "parse")

        with pytest.raises(ValueError, match="must be absent"):
            parse_linked_table(
                adata,
                TableLinkage(("cells",), "region", "instance"),
                context=_CONTEXT,
            )

        assert adata.X is expression
        assert adata.uns[TableModel.ATTRS_KEY] is stale
        parser.assert_not_called()

    @pytest.mark.parametrize(
        ("regions", "instances", "match"),
        [
            (["cells", "cells"], np.asarray([1, 1], dtype=np.int64), "duplicate"),
            (["cells", None], np.asarray([1, 2], dtype=np.int64), "null values"),
            (["cells", "other"], np.asarray([1, 2], dtype=np.int64), "observed regions"),
            ([1, 1], np.asarray([1, 2], dtype=np.int64), "string region"),
            (["cells", "cells"], np.asarray([1.0, 2.0]), "parser-compatible"),
            (["cells", "cells"], np.asarray([True, False]), "parser-compatible"),
            (["cells", "cells"], np.asarray([1, 2], dtype=np.int8), "parser-compatible"),
            (["cells", "cells"], np.asarray([1, 2], dtype=np.uint8), "parser-compatible"),
            (["cells", "cells"], np.asarray([1, "2"], dtype=object), "parser-compatible"),
            (["cells", "cells"], np.asarray([1, 2], dtype=object), "parser-compatible"),
            (["cells", "cells"], np.asarray([b"1", b"2"], dtype=object), "parser-compatible"),
            (["cells", "cells"], pd.Series([1, 2], dtype="Int64"), "parser-compatible"),
            (["cells", "cells"], pd.Series([1, 2], dtype="UInt64"), "parser-compatible"),
            (["cells", "cells"], pd.Series([1, None], dtype="Int64"), "null instance"),
        ],
    )
    def test_rejects_malformed_linkage_before_normalization(
        self,
        mocker: MockerFixture,
        regions: list[object],
        instances: pd.Series | np.ndarray,
        match: str,
    ) -> None:
        adata = _table(regions=regions, instances=instances, x=np.asarray([[1, 0], [0, 2]], dtype=np.int16))
        expression = adata.X
        normalization = mocker.patch("spatialdata_io.readers._utils.table._linkage.normalize_owned_anndata")
        parser = mocker.patch.object(TableModel, "parse")

        with pytest.raises(ReaderFormatError, match=match):
            parse_linked_table(
                adata,
                TableLinkage(("cells",), "region", "instance"),
                context=_CONTEXT,
            )

        assert adata.X is expression
        assert TableModel.ATTRS_KEY not in adata.uns
        normalization.assert_not_called()
        parser.assert_not_called()

    def test_rejects_missing_columns_before_normalization(self, mocker: MockerFixture) -> None:
        adata = _table()
        obs = cast("pd.DataFrame", adata.obs)
        del obs["instance"]
        normalization = mocker.patch("spatialdata_io.readers._utils.table._linkage.normalize_owned_anndata")

        with pytest.raises(ReaderFormatError, match="missing observation columns"):
            parse_linked_table(
                adata,
                TableLinkage(("cells",), "region", "instance"),
                context=_CONTEXT,
            )

        normalization.assert_not_called()

    @pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
    def test_accepts_parser_compatible_integer_widths(
        self,
        mocker: MockerFixture,
        dtype: np.dtype,
    ) -> None:
        adata = _table(instances=np.asarray([1, 2], dtype=dtype))
        mocker.patch.object(TableModel, "parse", side_effect=_public_parser_stub)

        result = parse_linked_table(
            adata,
            TableLinkage(("cells",), "region", "instance"),
            context=_CONTEXT,
        )

        assert result.obs["instance"].dtype == dtype

    @pytest.mark.parametrize(
        "instances",
        [
            pd.Series(["01", "02"], dtype="string"),
            pd.Series(np.asarray(["01", "02"], dtype=object), dtype=object),
            pd.Categorical(["01", "02"]),
            pd.Categorical(np.asarray([1, 2], dtype=np.uint16)),
        ],
    )
    def test_accepts_supported_string_and_categorical_storage(
        self,
        mocker: MockerFixture,
        instances: pd.Series | pd.Categorical,
    ) -> None:
        adata = _table(instances=instances)
        mocker.patch.object(TableModel, "parse", side_effect=_public_parser_stub)

        parse_linked_table(
            adata,
            TableLinkage(("cells",), "region", "instance"),
            context=_CONTEXT,
        )

    def test_allows_same_instance_in_different_regions(self, mocker: MockerFixture) -> None:
        adata = _table(regions=["a", "b"], instances=np.asarray([7, 7], dtype=np.int32))
        mocker.patch.object(TableModel, "parse", side_effect=_public_parser_stub)

        parse_linked_table(
            adata,
            TableLinkage(("a", "b"), "region", "instance"),
            context=_CONTEXT,
        )

    def test_preserves_parser_exception_identity_and_adds_context(self, mocker: MockerFixture) -> None:
        adata = _table()
        original = TypeError("injected parser failure")
        mocker.patch.object(TableModel, "parse", side_effect=original)

        with pytest.raises(TypeError, match="injected") as raised:
            parse_linked_table(
                adata,
                TableLinkage(("cells",), "region", "instance"),
                context=_CONTEXT,
            )

        assert raised.value is original
        assert "test_reader" in raised.value.__notes__[0]
        assert isinstance(adata.X, sparse.csr_array)


class TestExpectedInstanceIds:
    """Validate exact target comparison without semantic coercion."""

    def test_accepts_partial_reordered_targets_without_mutation(self, mocker: MockerFixture) -> None:
        adata = _table(
            regions=["a", "a"],
            instances=np.asarray([1, 2], dtype=np.uint32),
        )
        target = pd.Index(np.asarray([2, 1], dtype=np.int16), name="target")
        values_before = target.to_numpy(copy=True)
        mocker.patch.object(TableModel, "parse", side_effect=_public_parser_stub)

        parse_linked_table(
            adata,
            TableLinkage(("a",), "region", "instance"),
            context=_CONTEXT,
            expected_instance_ids={"a": target},
        )

        assert target.dtype == np.int16
        assert target.name == "target"
        np.testing.assert_array_equal(target, values_before)

    def test_compares_mixed_integer_widths_without_float_promotion(self, mocker: MockerFixture) -> None:
        maximum = np.iinfo(np.uint64).max
        adata = _table(
            regions=["a", "b"],
            instances=np.asarray([1, maximum], dtype=np.uint64),
        )
        targets = {
            "b": pd.Index(np.asarray([maximum], dtype=np.uint64)),
            "a": pd.Index(np.asarray([1], dtype=np.int64)),
        }
        mocker.patch.object(TableModel, "parse", side_effect=_public_parser_stub)

        parse_linked_table(
            adata,
            TableLinkage(("a", "b"), "region", "instance"),
            context=_CONTEXT,
            expected_instance_ids=targets,
        )

    def test_detects_uint64_adjacent_mismatch(self, mocker: MockerFixture) -> None:
        maximum = np.iinfo(np.uint64).max
        adata = _table(
            regions=["a", "b"],
            instances=np.asarray([1, maximum], dtype=np.uint64),
        )
        normalization = mocker.patch("spatialdata_io.readers._utils.table._linkage.normalize_owned_anndata")

        with pytest.raises(ReaderFormatError, match="1 target pairs missing"):
            parse_linked_table(
                adata,
                TableLinkage(("a", "b"), "region", "instance"),
                context=_CONTEXT,
                expected_instance_ids={
                    "a": pd.Index(np.asarray([1], dtype=np.int64)),
                    "b": pd.Index(np.asarray([maximum - 1], dtype=np.uint64)),
                },
            )

        normalization.assert_not_called()

    @pytest.mark.parametrize(
        ("target", "match"),
        [
            (pd.Index(np.asarray([-1, 2], dtype=np.int64)), "losslessly representable"),
            (pd.Index([True, False]), "non-boolean"),
            (pd.Index(["1", "2"]), "string target"),
            (pd.Index([1.0, 2.0]), "non-boolean"),
            (pd.Index([1, 1]), "duplicate"),
            (pd.Index([1, None]), "null"),
            (pd.Index([], dtype=object), "dtype object"),
            (pd.MultiIndex.from_tuples([("x", 1), ("x", 2)]), "2-level"),
        ],
    )
    def test_rejects_malformed_or_incompatible_targets(self, target: pd.Index, match: str) -> None:
        adata = _table(instances=np.asarray([1, 2], dtype=np.uint64))

        with pytest.raises(ReaderFormatError, match=match):
            parse_linked_table(
                adata,
                TableLinkage(("cells",), "region", "instance"),
                context=_CONTEXT,
                expected_instance_ids={"cells": target},
            )

    @pytest.mark.parametrize(
        "target",
        [
            pd.Index(np.asarray([2, 1], dtype=np.int8)),
            pd.Index(pd.array([2, 1], dtype="UInt64")),
            pd.Index(np.asarray([2, 1], dtype=object), dtype=object),
            pd.CategoricalIndex(pd.Index(np.asarray([2, 1], dtype=np.int16))),
        ],
    )
    def test_accepts_broader_lossless_integer_target_storage(
        self,
        mocker: MockerFixture,
        target: pd.Index,
    ) -> None:
        adata = _table(instances=np.asarray([1, 2], dtype=np.uint64))
        mocker.patch.object(TableModel, "parse", side_effect=_public_parser_stub)

        parse_linked_table(
            adata,
            TableLinkage(("cells",), "region", "instance"),
            context=_CONTEXT,
            expected_instance_ids={"cells": target},
        )

    def test_preserves_leading_zero_string_identity(self, mocker: MockerFixture) -> None:
        adata = _table(instances=pd.Series(["01", "1"], dtype="string"))
        mocker.patch.object(TableModel, "parse", side_effect=_public_parser_stub)

        parse_linked_table(
            adata,
            TableLinkage(("cells",), "region", "instance"),
            context=_CONTEXT,
            expected_instance_ids={"cells": pd.Index(["1", "01"])},
        )

    def test_rejects_mapping_key_outside_descriptor(self) -> None:
        adata = _table()

        with pytest.raises(ValueError, match="outside TableLinkage"):
            parse_linked_table(
                adata,
                TableLinkage(("cells",), "region", "instance"),
                context=_CONTEXT,
                expected_instance_ids={"other": pd.Index([1, 2])},
            )
