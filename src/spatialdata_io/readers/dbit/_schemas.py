"""DBiT positional barcode, imported H5AD, and final grid annotation schemas.

Barcode maps have exactly two positional string fields with unordered rows.
H5AD fields are named, unordered extensions whose storage/missingness remain
unchanged. Reader-created A/B and pixel identifiers use exact int64 Arrow
intermediates; sample strings are Arrow-backed and regions categorical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    import anndata as ad

    from spatialdata_io.readers._utils.paths import ArtifactFile

BARCODE_SOURCE_DTYPE = pd.ArrowDtype(pa.string())
GRID_DTYPE = pd.ArrowDtype(pa.int64())
SAMPLE_DTYPE = pd.ArrowDtype(pa.string())
_MAX_GRID_SIDE = 50
_BARCODE_COLUMNS = 2
BARCODE_LENGTH = 8


def _normalize_barcodes(frame: pd.DataFrame, path: ArtifactFile) -> None:
    """Validate bounded positional rows and normalize case without changing row order."""
    if frame.shape[1] != _BARCODE_COLUMNS or not 0 < len(frame) <= 2 * _MAX_GRID_SIDE:
        raise ReaderFormatError(
            context=ReaderErrorContext("dbit", "barcode positions"),
            path=path,
            found=f"barcode map shape {frame.shape}",
            expected="a nonempty headerless two-column TSV with at most fifty A and fifty B positions",
        )

    frame.columns = ["position", "sequence"]
    frame["sequence"] = frame["sequence"].str.upper()
    positions, sequences = frame["position"], frame["sequence"]
    if (
        positions.isna().any()
        or sequences.isna().any()
        or not positions.str.fullmatch(r"[AB]([1-9]|[1-4][0-9]|50)").all()
        or not sequences.str.fullmatch(r"[ACGT]{8}").all()
    ):
        raise ReaderFormatError(
            context=ReaderErrorContext("dbit", "barcode positions"),
            path=path,
            found="invalid barcode position or sequence",
            expected="positions A1..A50 or B1..B50 and eight A/C/G/T bases (case-insensitive)",
        )
    if not positions.is_unique:
        raise ReaderFormatError(
            context=ReaderErrorContext("dbit", "barcode positions"),
            path=path,
            found="duplicate positions",
            expected="one barcode per A/B position",
        )

    frame["axis"] = positions.str.slice(0, 1)
    frame["position"] = positions.str.slice(1).astype(GRID_DTYPE)

    # Sequence uniqueness is axis-local: the same DNA can identify one A and
    # one B position because its place in the combined observation ID differs.
    extents: list[int] = []
    for axis in ("A", "B"):
        selected = frame["axis"].eq(axis)
        axis_sequences = sequences[selected]
        numbers = frame.loc[selected, "position"]
        if not axis_sequences.is_unique or set(numbers) != set(range(1, len(numbers) + 1)):
            raise ReaderFormatError(
                context=ReaderErrorContext("dbit", "barcode positions"),
                path=path,
                found=f"duplicate barcodes or noncontiguous {axis} positions",
                expected="unique sequences within each axis and contiguous positions starting at one",
            )

        extents.append(len(numbers))

    # Equal contiguous axes make their length the side of the square grid.
    if not extents[0] or extents[0] != extents[1]:
        raise ReaderFormatError(
            context=ReaderErrorContext("dbit", "barcode positions"),
            path=path,
            found="unequal or missing grid axes",
            expected="the same nonempty A and B extent",
        )


def _validate_table(table: ad.AnnData, path: ArtifactFile) -> None:
    """Require nonempty expression and unique nonnull indexes and annotation names."""
    context = ReaderErrorContext("dbit", "expression table")
    if "spatial_grid" in table.obsm or POSITIONS_KEY in table.uns:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="occupied grid position destination",
            expected="no spatial_grid embedding or position declaration",
        )
    if "spatial" in table.obsm and "spatial_source" in table.obsm:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="occupied obsm[spatial_source]",
            expected="an unoccupied imported-coordinate destination",
        )
    if table.X is None or not table.n_obs or not table.n_vars:
        raise ReaderFormatError(
            context=context, path=path, found="empty or absent expression", expected="nonempty observation-by-feature X"
        )
    if not isinstance(table.obs, pd.DataFrame) or not isinstance(table.var, pd.DataFrame):
        message = "DBiT requires eager annotation DataFrames from read_h5ad."
        raise TypeError(message)

    if not table.obs.columns.is_unique or not table.var.columns.is_unique:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="duplicate annotation columns",
            expected="unique observation and feature column names",
        )
    if (
        table.obs_names.hasnans
        or not table.obs_names.is_unique
        or not table.var_names.is_unique
        or table.var_names.hasnans
        or (table.var_names == "").any()
    ):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="invalid identifiers",
            expected="unique observations and nonempty unique features",
        )


def _validate_barcode_prefixes(names: pd.Index, path: ArtifactFile) -> None:
    """Require one 16-base B/A prefix per observation; suffixes remain meaningful IDs."""
    context = ReaderErrorContext("dbit", "expression table")
    if not names.str.fullmatch(r"[ACGT]{16}").all() or not names.is_unique:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="malformed or duplicate barcode pairs",
            expected="one observation per 16-base B-then-A barcode prefix. Optional suffixes are retained",
        )


def _validate_mapped_positions(a: pd.Series, b: pd.Series, path: ArtifactFile) -> None:
    """Require both source barcode halves in the validated position map."""
    context = ReaderErrorContext("dbit", "expression table")
    if a.isna().any() or b.isna().any():
        raise ReaderFormatError(
            context=context,
            path=path,
            found="unknown observation barcodes",
            expected="every B/A barcode in the position map",
        )


def _validate_output_column(obs: pd.DataFrame, key: str, values: pd.Series, *, path: ArtifactFile) -> None:
    """Reject conflicting reserved metadata using exact scalar comparisons.

    Equal existing values are accepted; the generated field takes its declared
    output storage. Python scalar comparison avoids float promotion of uint64 IDs.
    """
    if key in obs and any(
        not pd.api.types.is_scalar(left)
        or pd.isna(left)
        or isinstance(left, bool) != isinstance(right, bool)
        or left != right
        for left, right in zip(obs[key].tolist(), values.tolist(), strict=True)
    ):
        raise _table_error(
            path,
            found=f"conflicting reserved observation column {key!r}",
            expected="absent or semantically equal reader output metadata",
        )


def _table_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a reserved output annotation conflict."""
    return ReaderFormatError(
        context=ReaderErrorContext("dbit", "expression table"), path=path, found=found, expected=expected
    )


def _source_linkage_columns(table: ad.AnnData, path: ArtifactFile) -> tuple[str, ...]:
    """Recognize only declared source region/pixel fields replaced by this reader."""
    previous = table.uns.get("spatialdata_attrs")
    if (
        not isinstance(previous, dict)
        or previous.get("region_key") != "region"
        or previous.get("instance_key") != "pixel_id"
    ):
        return ()
    regions = previous.get("region")
    if isinstance(regions, str):
        valid_regions = bool(regions)
    else:
        valid_regions = (
            isinstance(regions, (list, tuple, np.ndarray))
            and len(regions) > 0
            and all(isinstance(region, str) and bool(region) for region in regions)
        )
    if not valid_regions or not {"region", "pixel_id"}.issubset(table.obs.columns):
        raise _table_error(
            path,
            found="malformed previous source linkage",
            expected="a nonempty region declaration and its region/pixel_id columns",
        )
    return ("region", "pixel_id")
