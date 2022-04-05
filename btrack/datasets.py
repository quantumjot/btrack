import os
from typing import List

import numpy as np
import pooch
from skimage.io import imread

from .btypes import PyTrackObject
from .dataio import import_CSV

BASE_URL = (
    "https://raw.githubusercontent.com/lowe-lab-ucl/btrack-examples/main/"
)

CACHE_PATH = pooch.os_cache("btrack-examples")


def _remote_registry() -> os.PathLike:
    file_path = pooch.retrieve(
        # URL to one of Pooch's test files
        path=CACHE_PATH,
        url=BASE_URL + "registry.txt",
        known_hash="673de62c62eeb6f356fb1bff968748566d23936f567201cf61493d031d42d480",
    )
    return file_path


POOCH = pooch.create(
    path=CACHE_PATH,
    base_url=BASE_URL,
    version_dev="main",
    registry=None,
)
POOCH.load_registry(_remote_registry())


def cell_config() -> os.PathLike:
    """Return the file path to the example `cell_config`."""
    file_path = POOCH.fetch("examples/cell_config.json")
    return file_path


def particle_config() -> os.PathLike:
    """Return the file path to the example `particle_config`."""
    file_path = POOCH.fetch("examples/particle_config.json")
    return file_path


def example_segmentation_file() -> os.PathLike:
    """Return the file path to the example U-Net segmentation image file."""
    file_path = POOCH.fetch("examples/segmented.tif")
    return file_path


def example_segmentation() -> np.array:
    """Return the U-Net segmentation as a numpy array of dimensions (T, Y, X)."""
    file_path = example_segmentation_file()
    segmentation = imread(file_path)
    return segmentation


def example_track_objects_file() -> os.PathLike:
    """Return the file path to the example localized and classified objects
    stored in a CSV file."""
    file_path = POOCH.fetch("examples/objects.csv")
    return file_path


def example_track_objects() -> List[PyTrackObject]:
    """Return the example localized and classified objects stored in a CSV file
    as a list `PyTrackObject`s to be used by the tracker."""
    file_path = example_track_objects_file()
    objects = import_CSV(file_path)
    return objects
