"""Expected barcode-by-feature matrix and annotations shared by both 10x formats."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from spatialdata_io.readers._utils.errors import ReaderErrorContext

if TYPE_CHECKING:
    from anndata import AnnData

CONTEXT = ReaderErrorContext(reader="integration", artifact="10x counts", detected_version="synthetic")
BARCODES = ("barcode-b", "barcode-a", "barcode-c", "barcode-empty")
FEATURE_IDS = ("gene-id-a", "protein-id", "gene-id-b", "gene-id-empty")
FEATURE_NAMES = ("duplicate", "protein", "duplicate", "empty-gene")
FEATURE_TYPES = ("Gene Expression", "Antibody Capture", "Gene Expression", "Gene Expression")
COUNTS = sparse.csr_array(
    (
        np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int32),
        np.asarray([0, 1, 1, 2, 0, 2], dtype=np.int32),
        np.asarray([0, 2, 4, 6, 6], dtype=np.int32),
    ),
    shape=(4, 4),
)


def assert_canonical_sparse(table: AnnData) -> sparse.csr_array:
    assert not table.isbacked
    assert not table.is_view
    expression = table.X
    assert isinstance(expression, sparse.csr_array)
    assert isinstance(expression, sparse.sparray)
    assert not isinstance(expression, sparse.spmatrix)
    assert expression.has_canonical_format
    return expression


def assert_sparse_equal(actual: sparse.csr_array, expected: sparse.csr_array) -> None:
    assert actual.shape == expected.shape
    difference = actual - expected
    difference.eliminate_zeros()
    assert difference.nnz == 0
