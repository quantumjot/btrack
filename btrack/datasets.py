import os

import numpy as np
import pooch
from skimage.io import imread

from .btypes import PyTrackObject, Tracklet
from .io import HDF5FileHandler, import_CSV

BASE_URL = "https://raw.githubusercontent.com/lowe-lab-ucl/btrack-examples/main/"

CACHE_PATH = pooch.os_cache("btrack-examples")


def _remote_registry() -> os.PathLike:
    file_path = pooch.retrieve(
        # URL to one of Pooch's test files
        path=CACHE_PATH,
        url=BASE_URL + "registry.txt",
        known_hash="20d8c44289f421ab52d109e6af2c76610e740230479fe5c46a4e94463c9b5d50",
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


def example_segmentation() -> np.ndarray:
    """Return the U-Net segmentation as a numpy array of dimensions (T, Y, X)."""
    file_path = example_segmentation_file()
    segmentation = imread(file_path)
    return segmentation


def example_track_objects_file() -> os.PathLike:
    """Return the file path to the example localized and classified objects
    stored in a CSV file."""
    file_path = POOCH.fetch("examples/objects.csv")
    return file_path


def example_track_objects() -> list[PyTrackObject]:
    """Return the example localized and classified objects stored in a CSV file
    as a list `PyTrackObject`s to be used by the tracker."""
    file_path = example_track_objects_file()
    objects = import_CSV(file_path)
    return objects


def example_tracks() -> list[Tracklet]:
    """Return the example example localized and classified objected stored in an
    HDF5 file as a list of `Tracklet`s."""
    file_path = POOCH.fetch("examples/tracks.h5")
    with HDF5FileHandler(file_path, "r", obj_type="obj_type_1") as reader:
        tracks = reader.tracks
    return tracks
