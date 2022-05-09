from pathlib import Path

import pytest

import btrack.dataio

from ._utils import create_test_object


@pytest.fixture
def test_objects():
    """
    Create a list of 10 test objects.
    """
    return [create_test_object(id=i)[0] for i in range(10)]


@pytest.fixture
def hdf5_file_path(tmp_path, test_objects) -> Path:
    """
    Create and save a btrack HDF5 file, and return the path.

    Note that this only saves segmentation results, not tracking results.
    """
    fn = Path(tmp_path) / "test.h5"

    with btrack.dataio.HDF5FileHandler(fn, "w") as h:
        h.write_objects(test_objects)

    return fn
