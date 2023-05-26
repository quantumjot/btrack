import os
from typing import Union

import numpy as np
import pytest

import btrack

from ._utils import (
    RANDOM_SEED,
    TEST_DATA_PATH,
    create_test_object,
    create_test_segmentation_and_tracks,
)


@pytest.fixture
def test_objects():
    """
    Create a list of 10 test objects.
    """
    n_rows = 10
    return [create_test_object(test_id=i)[0] for i in range(n_rows)]


@pytest.fixture
def test_real_objects():
    """
    Create a list of objects from real data.
    """
    return btrack.io.import_CSV(TEST_DATA_PATH / "test_data.csv")


def write_h5_file(file_path: os.PathLike, test_objects) -> os.PathLike:
    """
    Write a h5 file with test objects and return path.
    """
    with btrack.io.HDF5FileHandler(file_path, "w") as h:
        h.write_objects(test_objects)

    return file_path


@pytest.fixture
def hdf5_file_path(tmp_path, test_objects) -> os.PathLike:
    """
    Create and save a btrack HDF5 file, and return the path.

    Note that this only saves segmentation results, not tracking results.
    """
    return write_h5_file(tmp_path / "test.h5", test_objects)


@pytest.fixture(params=["single", "list"])
def hdf5_file_path_or_paths(
    tmp_path, test_objects, request
) -> Union[os.PathLike, list[os.PathLike]]:
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
    else:
        raise ValueError(
            "Invalid requests.param, must be one of 'single' or 'list'"
        )


@pytest.fixture
def test_segmentation_and_tracks():
    """
    Create a test segmentation, ground truth and example tracks.
    """
    return create_test_segmentation_and_tracks(ndim=2)


@pytest.fixture
def default_rng():
    """
    Create a default PRNG to use for tests.
    """
    return np.random.default_rng(seed=RANDOM_SEED)
