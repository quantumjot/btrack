import warnings

warnings.warn(
    "`btrack.dataio` has been deprecated. Please use `btrack.io` subpackage "
    "instead.",
    # DeprecationWarning,
)

from .io import HDF5FileHandler  # noqa: F401,E402

if __name__ == "__main__":
    pass
