from .io import HDF5FileHandler  # noqa: F401

raise DeprecationWarning(
    "`btrack.dataio` has been deprecated. Please us `btrack.io` subpackage instead."
)


if __name__ == "__main__":
    pass
