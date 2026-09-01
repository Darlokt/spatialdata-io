"""Test the deliberately small shared CLI boundary."""

from __future__ import annotations

import click
import pytest

from spatialdata_io._cli._common import parse_json_object


class TestParseJsonObject:
    """Test the temporary structured-option parser used by legacy commands."""

    def test_returns_decoded_object(self) -> None:
        assert parse_json_object('{"chunks": [1, 2]}', parameter="options") == {"chunks": [1, 2]}

    def test_reports_parameter_and_decoder_failure(self) -> None:
        with pytest.raises(click.BadParameter, match=r"options.*Expecting property name"):
            parse_json_object("{invalid}", parameter="options")

    @pytest.mark.parametrize("value", ["[]", "null", "1", '"text"', "true"])
    def test_rejects_valid_non_object_json(self, value: str) -> None:
        with pytest.raises(click.BadParameter, match=r"options.*expected an object"):
            parse_json_object(value, parameter="options")
