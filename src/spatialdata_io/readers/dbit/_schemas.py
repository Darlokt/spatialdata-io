"""DBiT barcode-map, imported H5AD and generated grid-column schemas.

Barcode map (``barcode_list*``): a headerless, tab-separated file with exactly
two positional fields per row and no row-order requirement::

    A1 < TAB > AACCGGTT  # position: axis letter A or B and a one-based ordinal
    B1 < TAB > GGGGTTTT  # sequence: eight DNA bases, case-insensitive

- positions are ``A1..AN`` and ``B1..BN`` with the same contiguous extent and
  ``1 <= N <= 50``, each listed once;
- sequences are unique within an axis; one sequence may occur on both axes,
  because its place in an observation name tells the axes apart.

Imported H5AD (named fields, any order, extra fields kept):

- observation names start with a 16-base barcode pair, eight B bases followed
  by eight A bases (case-insensitive); any suffix is kept as part of the
  identifier, and the pairs are unique;
- feature names are nonempty and unique; ``obs`` and ``var`` column names are
  unique;
- ``X`` is present and the table has observations and features;
- ``obsm["spatial_grid"]`` and ``uns["spatialdata_io_positions"]`` are absent,
  since the reader writes them; an imported ``obsm["spatial"]`` moves to an
  unoccupied ``obsm["spatial_source"]``.

Generated observation columns use exact Arrow intermediates until the final
table conversion: ``array_A``, ``array_B`` and ``pixel_id`` are uint16 and
``sample`` is a string. ``region`` is categorical.

DBiT files carry no physical calibration, so no element uses ``global``
(micrometres). The grid is stored in ``grid_pixels``, one unit per barcode-grid
step (a DBiT pixel); an image is in ``image_pixels``, which the grid also
reaches through a scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    import anndata as ad

    from spatialdata_io.readers._utils.paths import ArtifactFile

BARCODE_TYPES = pa.schema([("position", pa.string()), ("sequence", pa.string())])
# The supported 50-by-50 grid has at most 2,500 spots; one unsigned type fits both ordinals and row IDs.
GRID_DTYPE = pd.ArrowDtype(pa.uint16())
SAMPLE_DTYPE = pd.ArrowDtype(pa.string())
BARCODE_LENGTH = 8
GRID_FRAME = "grid_pixels"
IMAGE_FRAME = "image_pixels"
MAX_GRID_SIDE = 50
_BARCODE_CONTEXT = ReaderErrorContext("dbit", "barcode positions")
_TABLE_CONTEXT = ReaderErrorContext("dbit", "expression table")


def normalize_barcodes(frame: pd.DataFrame, path: ArtifactFile) -> None:
    """Validate the positional barcode rows and normalize them in place.

    Storage parsing establishes the two named string columns. On return
    ``frame`` has the columns ``position`` (uint16 ordinal),
    ``sequence`` (upper case) and ``axis`` (``"A"`` or ``"B"``), in source row
    order.
    """
    if not 0 < len(frame) <= 2 * MAX_GRID_SIDE:
        raise ReaderFormatError(
            context=_BARCODE_CONTEXT,
            path=path,
            found=f"barcode map shape {frame.shape}",
            expected="a nonempty headerless two-column TSV with at most fifty A and fifty B positions",
        )

    frame["sequence"] = frame["sequence"].str.upper()
    positions, sequences = frame["position"], frame["sequence"]
    if (
        positions.isna().any()
        or sequences.isna().any()
        or not positions.str.fullmatch(r"[AB]([1-9]|[1-4][0-9]|50)").all()
        or not sequences.str.fullmatch(r"[ACGT]{8}").all()
    ):
        raise ReaderFormatError(
            context=_BARCODE_CONTEXT,
            path=path,
            found="invalid barcode position or sequence",
            expected="positions A1..A50 or B1..B50 and eight A/C/G/T bases (case-insensitive)",
        )
    if not positions.is_unique:
        raise ReaderFormatError(
            context=_BARCODE_CONTEXT, path=path, found="duplicate positions", expected="one barcode per A/B position"
        )

    frame["axis"] = positions.str.slice(0, 1)
    frame["position"] = positions.str.slice(1).astype(GRID_DTYPE)

    extents: list[int] = []
    for axis in ("A", "B"):
        selected = frame["axis"].eq(axis)
        numbers = frame.loc[selected, "position"]
        if not sequences[selected].is_unique or set(numbers) != set(range(1, len(numbers) + 1)):
            raise ReaderFormatError(
                context=_BARCODE_CONTEXT,
                path=path,
                found=f"duplicate barcodes or noncontiguous {axis} positions",
                expected="unique sequences within each axis and contiguous positions starting at one",
            )
        extents.append(len(numbers))

    # Equal contiguous axes make their length the side of the square grid.
    if not extents[0] or extents[0] != extents[1]:
        raise ReaderFormatError(
            context=_BARCODE_CONTEXT,
            path=path,
            found="unequal or missing grid axes",
            expected="the same nonempty A and B extent",
        )


def validate_table(table: ad.AnnData, path: ArtifactFile) -> None:
    """Validate imported identifiers, expression presence and coordinate destinations."""
    if "spatial_grid" in table.obsm or POSITIONS_KEY in table.uns:
        raise ReaderFormatError(
            context=_TABLE_CONTEXT,
            path=path,
            found="occupied grid position destination",
            expected="no spatial_grid embedding or position declaration",
        )
    if "spatial" in table.obsm and "spatial_source" in table.obsm:
        raise ReaderFormatError(
            context=_TABLE_CONTEXT,
            path=path,
            found="occupied obsm[spatial_source]",
            expected="an unoccupied imported-coordinate destination",
        )
    if table.X is None or not table.n_obs or not table.n_vars:
        raise ReaderFormatError(
            context=_TABLE_CONTEXT,
            path=path,
            found="empty or absent expression",
            expected="nonempty observation-by-feature X",
        )
    if not isinstance(table.obs, pd.DataFrame) or not isinstance(table.var, pd.DataFrame):
        message = "DBiT requires eager annotation DataFrames from read_h5ad."
        raise TypeError(message)

    if not table.obs.columns.is_unique or not table.var.columns.is_unique:
        raise ReaderFormatError(
            context=_TABLE_CONTEXT,
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
            context=_TABLE_CONTEXT,
            path=path,
            found="invalid identifiers",
            expected="unique observations and nonempty unique features",
        )


def validate_barcode_prefixes(names: pd.Index, path: ArtifactFile) -> None:
    """Require one unique upper-case 16-base B/A prefix per observation."""
    if not names.str.fullmatch(r"[ACGT]{16}").all() or not names.is_unique:
        raise ReaderFormatError(
            context=_TABLE_CONTEXT,
            path=path,
            found="malformed or duplicate barcode pairs",
            expected="one observation per 16-base B-then-A barcode prefix. Optional suffixes are retained",
        )


def validate_mapped_positions(a: pd.Series, b: pd.Series, path: ArtifactFile) -> None:
    """Require both barcode halves of every observation in the position map."""
    if a.isna().any() or b.isna().any():
        raise ReaderFormatError(
            context=_TABLE_CONTEXT,
            path=path,
            found="unknown observation barcodes",
            expected="every B/A barcode in the position map",
        )
