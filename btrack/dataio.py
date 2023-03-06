import warnings

warnings.warn(
    "`btrack.dataio` has been deprecated. Please use `btrack.io` subpackage "
    "instead.",
    # DeprecationWarning,
)

from btrack.io.hdf import HDF5FileHandler  # noqa: F401,E402

if __name__ == "__main__":
    pass
