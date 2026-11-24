"""QPI XML and acquisition profile metadata contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers.phenocycler._metadata import (
    ProfileChannel,
    description_version,
    fusion_protocol,
    match_profile,
    nominal_pixel_size_um,
    page_quantities,
    parse_description,
    profile_channels,
    scan_generation,
)
from tests.integration.readers.phenocycler._fixtures import CIM_PROFILE, FUSION_PROFILE

if TYPE_CHECKING:
    from pathlib import Path

_CONTEXT = ReaderErrorContext("phenocycler", "QPI scan")
_ROOT = "PerkinElmer-QPI-ImageDescription"


def _description(content: str) -> str:
    return f"<{_ROOT}><ImageType>FullResolution</ImageType>{content}</{_ROOT}>"


class TestParseDescription:
    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("<invalid", "malformed ImageDescription XML"),
            ("<Other/>", "unrecognized"),
            (f"<{_ROOT}><ImageType>Raw</ImageType></{_ROOT}>", "unrecognized"),
            (f"<!DOCTYPE foo [<!ENTITY x 'ignored'>]>{_description('')}", "document type"),
        ],
    )
    def test_bad_description_is_contextual(self, tmp_path: Path, text: str, message: str) -> None:
        path = tmp_path / "bad.qptiff"
        with pytest.raises(ReaderFormatError, match=message) as exc:
            parse_description(text, path=path, page_index=1, context=_CONTEXT)
        assert exc.value.path == path
        assert "IFD 1" in str(exc.value)


class TestProfileChannels:
    def test_cim_cycles_wavelengths_and_resolution(self, tmp_path: Path) -> None:
        description = _description(f"<ScanProfile>{CIM_PROFILE}</ScanProfile>")
        document, _ = parse_description(description, context=_CONTEXT, path=tmp_path, page_index=0)
        assert scan_generation(document) == "cim_qptiff"
        assert nominal_pixel_size_um(document) == 0.5
        assert description_version(document) is None
        assert profile_channels(document, context=_CONTEXT, path=tmp_path) == (
            ProfileChannel("DAPI", 1, 1, 20.0, 458.0),
            ProfileChannel("CD3", 1, 2, 100.0, 520.0),
        )

    def test_fusion_preserves_one_based_order_after_excluding_blanks(self, tmp_path: Path) -> None:
        description = _description(f"<ScanProfile>{FUSION_PROFILE}</ScanProfile>")
        document, _ = parse_description(description, context=_CONTEXT, path=tmp_path, page_index=0)
        assert profile_channels(document, context=_CONTEXT, path=tmp_path) == (
            ProfileChannel("DAPI", 1, 1, 88.0),
            ProfileChannel("CD3", 2, 1, 120.0),
        )


class TestFusionProtocol:
    def test_well_and_slot_positions_survive_invalid_and_blank_items(self) -> None:
        content: dict[str, object] = {
            "formatVersion": 3,
            "wells": [
                None,
                {"items": [None, {"markerName": "BLANK"}, {"markerName": " CD3 ", "exposure": True}]},
            ],
        }
        protocol = fusion_protocol(content)
        assert protocol.version == 3
        assert protocol.cycles == 2
        assert protocol.channels == (ProfileChannel("CD3", 2, 3, None),)


class TestMatchProfile:
    def test_match_by_marker_then_occurrence_leaves_missing_markers_unassigned(self) -> None:
        first = ProfileChannel("CD3", 1, 2, 10)
        second = ProfileChannel("CD3", 2, 1, 20)
        third = ProfileChannel("F4/80", 1, 3, 30)
        extra = ProfileChannel("F4/80", 2, 3, 30)
        assert match_profile(("CD3", "F4-80", "CD3", "DAPI"), (third, first, second)) == (first, third, second, None)
        assert match_profile(("F4-80",), (third, extra)) == (None,)

    def test_ambiguous_punctuation_keeps_exact_matches(self) -> None:
        first = ProfileChannel("F4/80", 1, 2, 10)
        second = ProfileChannel("F480", 2, 1, 20)
        assert match_profile(("F4-80", "F480"), (first, second)) == (None, second)


class TestPageQuantities:
    def test_limits_and_optional_values(self, tmp_path: Path) -> None:
        body = (
            "<Color>4,10,999</Color><SignalUnits>256</SignalUnits><ExposureTime>5.5</ExposureTime>"
            "<ScaleFactor>inf</ScaleFactor><NormalizedUnitsFactor>NULL</NormalizedUnitsFactor>"
            "<AutofluorescenceSubtracted>Unknown</AutofluorescenceSubtracted>"
        )
        document, _ = parse_description(_description(body), context=_CONTEXT, path=tmp_path, page_index=0)
        assert page_quantities(document) == {}
