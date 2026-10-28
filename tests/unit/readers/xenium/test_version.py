"""Map experiment.xenium format versions to bundle families; software strings never select a layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.xenium._version import FormatVersion, detect_version

PATH = Path("bundle/experiment.xenium")


class TestDetectVersion:
    """experiment.xenium versions select exactly one supported bundle family."""

    @pytest.mark.parametrize(
        ("fields", "family", "text"),
        [
            ({"major_version": 1, "minor_version": 1}, "integer_ids", "1.1"),
            ({"major_version": 2, "minor_version": 0}, "hex_ids", "2.0"),
            ({"major_version": 2, "minor_version": 0, "patch_version": 1}, "hex_ids", "2.0.1"),
            ({"major_version": 3, "minor_version": 0, "patch_version": 0}, "hex_ids", "3.0.0"),
            ({"major_version": 4, "minor_version": 0, "patch_version": 0}, "polygon_sets", "4.0.0"),
            ({"major_version": 5, "minor_version": 2, "patch_version": 0}, "polygon_sets", "5.2.0"),
            ({"major_version": 6, "minor_version": 1, "patch_version": 0}, "polygon_sets", "6.1.0"),
            ({"major_version": 6, "minor_version": 9, "patch_version": 7}, "polygon_sets", "6.9.7"),
        ],
    )
    def test_documented_format_versions(self, fields: dict[str, object], family: str, text: str) -> None:
        version, detected = detect_version(fields, "R&D", path=PATH)

        assert detected == family
        assert str(version) == text

    @pytest.mark.parametrize("software", ["Xenium-1.0.1", "xenium-1.0.2", "xenium-1.1.0.4", "xenium-1.0.1-3-gabc"])
    def test_unversioned_xoa_1_0_and_1_1(self, software: str) -> None:
        assert detect_version({}, software, path=PATH) == (None, "integer_ids")

    @pytest.mark.parametrize("software", ["xenium-1.2.0.1", "xenium-2.0.0.10", "R&D", "Development", ""])
    def test_missing_fields_need_xoa_1_0_or_1_1(self, software: str) -> None:
        with pytest.raises(ReaderFormatError, match=r"only XOA 1\.0/1\.1 bundles omit"):
            detect_version({}, software, path=PATH)

    def test_ranger_software_never_selects_the_layout(self) -> None:
        # A Ranger rename keeps the input format; only the version fields decide.
        experiment = {"major_version": 3, "minor_version": 0, "xenium_ranger": {"version": "xenium-3.1.0.5"}}

        assert detect_version(experiment, "xenium-1.9.0.1", path=PATH)[1] == "hex_ids"

    @pytest.mark.parametrize("major", [0, 7, 12])
    def test_unsupported_major_versions(self, major: int) -> None:
        with pytest.raises(ReaderFormatError, match=f"unsupported format major version {major}") as caught:
            detect_version({"major_version": major, "minor_version": 0}, "xenium-9.0.0.1", path=PATH)

        assert caught.value.context.detected_version == f"{major}.0"

    @pytest.mark.parametrize(
        "fields",
        [{"major_version": 4}, {"minor_version": 0}, {"patch_version": 0}],
        ids=["no-minor", "no-major", "patch-only"],
    )
    def test_partial_versions(self, fields: dict[str, object]) -> None:
        with pytest.raises(ReaderFormatError, match="partial format version"):
            detect_version(fields, "xenium-2.0.0.10", path=PATH)

    @pytest.mark.parametrize("value", [True, -1, 4.0, "4", None])
    def test_malformed_fields(self, value: object) -> None:
        experiment = {"major_version": value, "minor_version": 0}
        if value is None:
            with pytest.raises(ReaderFormatError, match="partial format version"):
                detect_version(experiment, "xenium-2.0.0.10", path=PATH)
        else:
            with pytest.raises(ReaderFormatError, match="non-negative integer format version field"):
                detect_version(experiment, "xenium-2.0.0.10", path=PATH)


class TestFormatVersion:
    """Format versions print as recorded."""

    def test_string_omits_an_absent_patch(self) -> None:
        assert str(FormatVersion(2, 0, None)) == "2.0"
        assert str(FormatVersion(6, 1, 0)) == "6.1.0"
