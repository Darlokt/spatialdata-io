"""Fixtures shared by command-boundary tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


@pytest.fixture
def reader_result(mocker: MockerFixture) -> MagicMock:
    """Return a result object that records the output-store write."""
    return mocker.MagicMock()
