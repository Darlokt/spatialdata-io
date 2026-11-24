"""Read processed Akoya PhenoCycler and PerkinElmer QPI scans.

The public entry point :func:`phenocycler` accepts an individual scan or a
directory of regions and returns their full-resolution measurements as images.
"""

from __future__ import annotations

from spatialdata_io.readers.phenocycler._reader import phenocycler

__all__ = ["phenocycler"]
