"""Read processed Akoya PhenoCycler and PerkinElmer QPI scans.

The public entry point :func:`phenocycler` accepts an individual scan or a
directory of regions and returns full-resolution measurements and any explicitly
associated instance masks.
"""

from __future__ import annotations

from spatialdata_io.readers.phenocycler._reader import phenocycler

__all__ = ["phenocycler"]
