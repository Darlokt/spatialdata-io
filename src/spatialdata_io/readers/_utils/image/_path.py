from __future__ import annotations

from pathlib import Path

from spatialdata_io.readers._utils.errors import RasterFormatError


def normalize_file_path(path: str | Path) -> Path:
    """Return a resolved regular file without interpreting its file name."""
    normalized = Path(path).expanduser().resolve(strict=True)
    if not normalized.is_file():
        message = f"Expected raster source to be a regular file: {normalized}"
        raise RasterFormatError(message)
    return normalized
