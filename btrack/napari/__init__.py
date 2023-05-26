try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

import logging

from btrack.napari import constants, main

__all__ = [
    "constants",
    "main",
]
