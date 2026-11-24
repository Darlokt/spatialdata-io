"""The registration conditions for QPI bit-depth companions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import require_file
from spatialdata_io.readers.phenocycler._images import inspect_scan
from spatialdata_io.readers.phenocycler._transforms import validate_companion
from tests.integration.readers.phenocycler._fixtures import CIM_PROFILE, write_qptiff

if TYPE_CHECKING:
    from pathlib import Path

_CONTEXT = ReaderErrorContext("phenocycler", "QPI scan")


class TestValidateCompanion:
    @pytest.mark.parametrize(
        ("field", "key"),
        [
            ("barcode", "profile_reagent_barcode"),
            ("clone", "profile_clone"),
            ("reporter", "profile_reporter"),
            ("dye", "profile_dye"),
        ],
    )
    def test_same_marker_and_band_with_conflicting_fusion_reagents(self, tmp_path: Path, field: str, key: str) -> None:
        high, low = tmp_path / "x.er.qptiff", tmp_path / "x.qptiff"
        for file, dtype, value in ((high, "uint16", "A"), (low, "uint8", "B")):
            profile = {"experimentDescription": {"wells": [{"items": [{"markerName": "CD3", field: value}]}]}}
            write_qptiff(file, dtype=dtype, scan_profile=json.dumps(profile))
        primary = inspect_scan(require_file(high, context=_CONTEXT), context=_CONTEXT)
        companion = inspect_scan(require_file(low, context=_CONTEXT), context=_CONTEXT)
        with pytest.raises(ReaderFormatError, match=key) as exc:
            validate_companion(primary, companion, context=_CONTEXT)
        assert exc.value.path == low
        assert "CD3" in exc.value.found

    @pytest.mark.parametrize(
        ("original", "replacement", "key"),
        [
            ("<Index>1</Index>", "<Index>3</Index>", "profile_cycle"),
            ("<Index>2</Index>", "<Index>3</Index>", "profile_channel"),
            ("<MarkerClass>Nuclear</MarkerClass>", "<MarkerClass>Membrane</MarkerClass>", "profile_marker_class"),
        ],
    )
    def test_conflicting_cim_channel_identity(self, tmp_path: Path, original: str, replacement: str, key: str) -> None:
        high, low = tmp_path / "x.er.qptiff", tmp_path / "x.qptiff"
        profile = CIM_PROFILE.replace("<MarkerName>DAPI", "<MarkerClass>Nuclear</MarkerClass><MarkerName>DAPI")
        write_qptiff(high, software="CODEX Instrument Manager", scan_profile=profile)
        write_qptiff(
            low,
            software="CODEX Instrument Manager",
            dtype="uint8",
            scan_profile=profile.replace(original, replacement, 1),
        )
        primary = inspect_scan(require_file(high, context=_CONTEXT), context=_CONTEXT)
        companion = inspect_scan(require_file(low, context=_CONTEXT), context=_CONTEXT)
        with pytest.raises(ReaderFormatError, match=key):
            validate_companion(primary, companion, context=_CONTEXT)

    @pytest.mark.parametrize(("high_barcode", "low_barcode"), [("BX001", "BX001"), ("BX001", None), (None, "BX001")])
    def test_matching_and_one_sided_reagent_identity(
        self, tmp_path: Path, high_barcode: str | None, low_barcode: str | None
    ) -> None:
        high, low = tmp_path / "x.er.qptiff", tmp_path / "x.qptiff"
        for file, dtype, barcode in ((high, "uint16", high_barcode), (low, "uint8", low_barcode)):
            profile = {"experimentDescription": {"wells": [{"items": [{"markerName": "CD3", "barcode": barcode}]}]}}
            write_qptiff(file, dtype=dtype, scan_profile=json.dumps(profile))
        primary = inspect_scan(require_file(high, context=_CONTEXT), context=_CONTEXT)
        companion = inspect_scan(require_file(low, context=_CONTEXT), context=_CONTEXT)
        validate_companion(primary, companion, context=_CONTEXT)

    @pytest.mark.parametrize(
        ("dtype", "names", "resolution_unit", "failure"),
        [
            ("uint16", ("DAPI", "CD3"), "CENTIMETER", "unsigned 8-bit"),
            ("uint8", ("DAPI", "ATTO 550"), "CENTIMETER", "different channel or spatial grid"),
            ("uint8", ("DAPI", "CD3"), "INCH", "different channel or spatial grid"),
        ],
    )
    def test_requires_measured_dtype_channel_and_grid(
        self, tmp_path: Path, dtype: str, names: tuple[str, str], resolution_unit: str, failure: str
    ) -> None:
        high = tmp_path / "x.er.qptiff"
        low = tmp_path / "x.qptiff"
        write_qptiff(high)
        write_qptiff(low, dtype=dtype, names=names, resolution_unit=resolution_unit)
        primary = inspect_scan(require_file(high, context=_CONTEXT), context=_CONTEXT)
        companion = inspect_scan(require_file(low, context=_CONTEXT), context=_CONTEXT)
        with pytest.raises(ReaderFormatError, match=failure):
            validate_companion(primary, companion, context=_CONTEXT)
