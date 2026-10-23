"""Exact annotation-storage conversion at the AnnData boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.table import prepare_anndata_dataframe

_CONTEXT = ReaderErrorContext(reader="example", artifact="annotations", detected_version="1")
_INTEGER_TYPES = [pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]


class TestPrepareAnndataDataframe:
    """Preserve exact values while replacing only incompatible storage."""

    @pytest.mark.parametrize("arrow_type", _INTEGER_TYPES)
    @pytest.mark.parametrize("nullable", [False, True])
    def test_integer_width_signedness_extrema_and_missingness(self, arrow_type: pa.DataType, *, nullable: bool) -> None:
        """Preserve integer extrema, widths, signedness, and nullable masks exactly."""
        dtype = np.dtype(str(arrow_type))
        limits = np.iinfo(dtype)
        values = [limits.max, limits.min]
        source = [*values, None] if nullable else values
        frame = pd.DataFrame({"id": pd.Series(source, dtype=pd.ArrowDtype(arrow_type)).array})

        result = prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert result is frame
        if nullable:
            expected_dtype = f"{'UInt' if pa.types.is_unsigned_integer(arrow_type) else 'Int'}{arrow_type.bit_width}"
            assert str(result["id"].dtype) == expected_dtype
            assert result["id"].iloc[-1] is pd.NA
        else:
            assert result["id"].dtype == dtype

        assert result["id"].iloc[:2].tolist() == values

    @pytest.mark.parametrize("nullable", [False, True])
    def test_boolean_storage(self, *, nullable: bool) -> None:
        """Retain boolean values and missingness in supported extension storage."""
        values = [True, False, None] if nullable else [True, False]
        frame = pd.DataFrame({"flag": pd.Series(values, dtype=pd.ArrowDtype(pa.bool_())).array})

        prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert frame["flag"].dtype == (pd.BooleanDtype() if nullable else np.dtype(np.bool_))
        assert frame["flag"].iloc[:2].tolist() == [True, False]

        if nullable:
            assert frame["flag"].iloc[2] is pd.NA

    @pytest.mark.parametrize("arrow_type", [pa.float32(), pa.float64()])
    def test_float_valid_nan_signed_zero_infinity_and_precision(self, arrow_type: pa.DataType) -> None:
        """Preserve floating values and valid NaN bits without changing width."""
        dtype = np.dtype(f"float{arrow_type.bit_width}")
        expected = np.asarray(
            [0.0, -0.0, np.nextafter(dtype.type(1), dtype.type(2)), np.inf, -np.inf, np.nan], dtype=dtype
        )
        arrow = pa.array(expected, type=arrow_type, from_pandas=False)
        frame = pd.DataFrame({"value": pd.Series(arrow, dtype=pd.ArrowDtype(arrow_type)).array})

        prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert frame["value"].dtype == dtype
        np.testing.assert_array_equal(
            frame["value"].to_numpy().view(f"uint{arrow_type.bit_width}"), expected.view(f"uint{arrow_type.bit_width}")
        )

    @pytest.mark.parametrize("arrow_type", [pa.float32(), pa.float64()])
    def test_float_nulls_require_explicit_semantic_permission(self, arrow_type: pa.DataType) -> None:
        """Require schema authorization before replacing Arrow nulls with NaN."""
        source = pa.array([1.25, None, np.nan], type=arrow_type, from_pandas=False)
        frame = pd.DataFrame({"value": pd.Series(source, dtype=pd.ArrowDtype(arrow_type)).array})
        np.testing.assert_array_equal(frame["value"].isna(), [False, True, False])

        with pytest.raises(ReaderFormatError, match="explicit schema authorization") as error:
            prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert error.value.context is _CONTEXT
        assert "column 'value'" in str(error.value)
        assert isinstance(frame["value"].dtype, pd.ArrowDtype)

        prepare_anndata_dataframe(frame, context=_CONTEXT, allow_float_nulls=True)

        assert frame["value"].dtype == np.dtype(f"float{arrow_type.bit_width}")
        np.testing.assert_array_equal(frame["value"].to_numpy(), [1.25, np.nan, np.nan])

    @pytest.mark.parametrize("arrow_type", [pa.string(), pa.large_string()])
    def test_strings_preserve_missingness_lexical_ids_and_unicode(self, arrow_type: pa.DataType) -> None:
        """Keep lexical identifiers, Unicode, whitespace, empty strings, and nulls."""
        frame = pd.DataFrame({"text": pd.Series(["001", "  β ", "", None], dtype=pd.ArrowDtype(arrow_type)).array})

        prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert frame["text"].dtype == pd.StringDtype(storage="pyarrow")
        assert frame["text"].tolist() == ["001", "  β ", "", pd.NA]

    def test_compatible_columns_and_index_keep_their_arrays(self) -> None:
        """Reuse compatible annotation arrays while replacing an Arrow column."""
        frame = pd.DataFrame(
            {
                "integer": np.asarray([1, 2], dtype=np.int16),
                "nullable": pd.Series([1, None], dtype="UInt64").array,
                "string": pd.Series(["a", None], dtype=pd.StringDtype(storage="pyarrow")).array,
                "category": pd.Categorical(["b", "a"], categories=["a", "b", "unused"], ordered=True),
                "object": pd.Series([object(), object()], dtype=object),
                "converted": pd.Series([3, 4], dtype="int32[pyarrow]").array,
            }
        )
        unchanged = {name: frame[name].array for name in ["nullable", "string", "category", "object"]}
        numeric = frame["integer"].to_numpy(copy=False)
        index = frame.index

        prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert frame.index is index
        for name, array in unchanged.items():
            if name == "object":
                assert np.shares_memory(frame[name].to_numpy(copy=False), array.to_numpy(copy=False))
            else:
                assert frame[name].array is array

        assert np.shares_memory(frame["integer"].to_numpy(copy=False), numeric)

    @pytest.mark.parametrize("arrow_type", [pa.int32(), pa.uint64(), pa.string()])
    def test_index_names_order_and_missingness(self, arrow_type: pa.DataType) -> None:
        """Convert Arrow indexes without changing names, order, or null values."""
        values = ["002", "001", None] if pa.types.is_string(arrow_type) else [2, 1, None]
        index = pd.Index(pd.Series(values, dtype=pd.ArrowDtype(arrow_type)).array, name="source_id")
        frame = pd.DataFrame({"value": [3, 4, 5]}, index=index)

        prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert frame.index.name == "source_id"
        assert frame.index.tolist() == [*values[:2], pd.NA]
        assert not isinstance(frame.index.dtype, pd.ArrowDtype)
        assert frame["value"].tolist() == [3, 4, 5]

    def test_multiindex_levels_and_codes(self) -> None:
        """Replace Arrow levels while preserving compound index membership and order."""
        index = pd.MultiIndex.from_arrays(
            [pd.Series([2, 1, 2], dtype="int16[pyarrow]").array, ["b", "a", "a"]], names=["integer", "text"]
        )
        expected = index.tolist()
        frame = pd.DataFrame(index=index)

        prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert isinstance(frame.index, pd.MultiIndex)
        assert frame.index.tolist() == expected
        assert frame.index.names == index.names
        assert frame.index.levels[0].dtype == np.dtype(np.int16)
        assert frame.index.levels[1] is index.levels[1] or frame.index.levels[1].equals(index.levels[1])

    def test_compatible_multiindex_is_reused(self) -> None:
        """Return an already compatible compound index unchanged."""
        index = pd.MultiIndex.from_tuples([(1, "b"), (2, "a")])
        frame = pd.DataFrame(index=index)

        prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert frame.index is index

    @pytest.mark.parametrize(
        "arrow_type",
        [
            pa.decimal128(12, 2),
            pa.list_(pa.int32()),
            pa.binary(),
            pa.timestamp("ms"),
            pa.date32(),
            pa.null(),
            pa.float16(),
        ],
    )
    def test_unsupported_arrow_storage_fails_contextually(self, arrow_type: pa.DataType) -> None:
        """Reject unsupported Arrow representations with artifact and column context."""
        frame = pd.DataFrame({"unsupported": pd.Series([None], dtype=pd.ArrowDtype(arrow_type)).array})

        with pytest.raises(ReaderFormatError, match="column 'unsupported'") as error:
            prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert error.value.context is _CONTEXT
        assert str(arrow_type) in str(error.value)

    def test_unsupported_index_reports_its_name(self) -> None:
        """Include the index name when its Arrow storage is unsupported."""
        frame = pd.DataFrame(
            index=pd.Index(pd.Series([None], dtype=pd.ArrowDtype(pa.binary())).array, name="bad_index")
        )

        with pytest.raises(ReaderFormatError, match="index 'bad_index'"):
            prepare_anndata_dataframe(frame, context=_CONTEXT)

    @pytest.mark.parametrize("arrow_type", [*_INTEGER_TYPES, pa.bool_(), pa.float32(), pa.float64(), pa.string()])
    def test_empty_columns_keep_declared_types(self, arrow_type: pa.DataType) -> None:
        """Retain declared semantic types when columns contain no rows."""
        frame = pd.DataFrame({"empty": pd.Series([], dtype=pd.ArrowDtype(arrow_type)).array})
        expected = (
            pd.StringDtype(storage="pyarrow")
            if pa.types.is_string(arrow_type)
            else np.dtype(arrow_type.to_pandas_dtype())
        )

        prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert frame.empty
        assert frame["empty"].dtype == expected

    def test_duplicate_column_labels_do_not_overwrite_other_positions(self) -> None:
        """Replace duplicate-named columns independently by their physical positions."""
        frame = pd.DataFrame(
            {"a": pd.Series([1], dtype="int16[pyarrow]").array, "b": pd.Series([2], dtype="uint32[pyarrow]").array}
        )
        frame.columns = pd.Index(["duplicate", "duplicate"])

        prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert frame.columns.tolist() == ["duplicate", "duplicate"]
        assert frame.iloc[:, 0].dtype == np.dtype(np.int16)
        assert frame.iloc[:, 1].dtype == np.dtype(np.uint32)
        np.testing.assert_array_equal(frame.to_numpy(), [[1, 2]])

    def test_chunked_arrow_exact_uint64(self) -> None:
        """Preserve uint64 identifiers across multiple source Arrow chunks."""
        source = pa.chunked_array([[2**64 - 1], [2**63 + 1]], type=pa.uint64())
        frame = pd.DataFrame({"id": pd.Series(source, dtype=pd.ArrowDtype(pa.uint64())).array})

        prepare_anndata_dataframe(frame, context=_CONTEXT)

        assert frame["id"].dtype == np.dtype(np.uint64)
        assert frame["id"].tolist() == [2**64 - 1, 2**63 + 1]
