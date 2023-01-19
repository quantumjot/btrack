import warnings

from .io import HDF5FileHandler  # noqa: F401

raise warnings.warn(
    "`btrack.dataio` has been deprecated. Please us `btrack.io` subpackage instead.",
    DeprecationWarning,
)


if __name__ == "__main__":
    pass
