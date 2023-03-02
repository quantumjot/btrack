import os

import pooch
from numpy import typing as npt
from skimage.io import imread

from btrack.btypes import PyTrackObject
from btrack.io.importers import import_CSV

BASE_URL = "https://raw.githubusercontent.com/lowe-lab-ucl/btrack-examples/main/"

CACHE_PATH = pooch.os_cache("btrack-examples")


def _remote_registry() -> os.PathLike:
    return pooch.retrieve(
        path=CACHE_PATH,
        url=f"{BASE_URL}registry.txt",
        known_hash="673de62c62eeb6f356fb1bff968748566d23936f567201cf61493d031d42d480",
    )


POOCH = pooch.create(
    path=CACHE_PATH,
    base_url=BASE_URL,
    version_dev="main",
    registry=None,
)
POOCH.load_registry(_remote_registry())


def cell_config() -> os.PathLike:
    """Return the file path to the example `cell_config`."""
    return POOCH.fetch("examples/cell_config.json")


def particle_config() -> os.PathLike:
    """Return the file path to the example `particle_config`."""
    return POOCH.fetch("examples/particle_config.json")


def example_segmentation_file() -> os.PathLike:
    """Return the file path to the example U-Net segmentation image file."""
    return POOCH.fetch("examples/segmented.tif")


def example_segmentation() -> npt.NDArray:
    """Return the U-Net segmentation as a numpy array of dimensions (T, Y, X)."""
    file_path = example_segmentation_file()
    return imread(file_path)


def example_track_objects_file() -> os.PathLike:
    """Return the file path to the example localized and classified objects
    stored in a CSV file."""
    return POOCH.fetch("examples/objects.csv")


def example_track_objects() -> list[PyTrackObject]:
    """Return the example localized and classified objects stored in a CSV file
    as a list `PyTrackObject`s to be used by the tracker."""
    file_path = example_track_objects_file()
    return import_CSV(file_path)
