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


def write_h5_file(file_path: Path, test_objects) -> Path:
    """
    Write a h5 file with test objects and return path.
    """
    with btrack.dataio.HDF5FileHandler(file_path, "w") as h:
        h.write_objects(test_objects)

    return file_path


@pytest.fixture
def hdf5_file_path(tmp_path, test_objects) -> Path:
    """
    Create and save a btrack HDF5 file, and return the path.

    Note that this only saves segmentation results, not tracking results.
    """
    return write_h5_file(tmp_path / "test.h5", test_objects)


@pytest.fixture(params=["single", "list"])
def hdf5_file_path_or_paths(tmp_path, test_objects, request) -> Path:
    """
    Create and save a btrack HDF5 file, and return the path.

    Note that this only saves segmentation results, not tracking results.
    """
    if request.param == "single":
        return write_h5_file(tmp_path / "test.h5", test_objects)
    elif request.param == "list":
        return [
            write_h5_file(tmp_path / "test1.h5", test_objects),
            write_h5_file(tmp_path / "test2.h5", test_objects),
        ]
