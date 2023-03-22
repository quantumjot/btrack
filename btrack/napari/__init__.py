try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

import logging

from napari_btrack import constants, main

__all__ = [
    "constants",
    "main",
]
