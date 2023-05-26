import warnings

warnings.warn(  # noqa: B028
    "`btrack.dataio` has been deprecated. Please use `btrack.io` subpackage "
    "instead.",
    # DeprecationWarning,
)

from .io import HDF5FileHandler  # noqa: F401,E402
