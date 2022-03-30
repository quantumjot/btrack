import os

import pooch

BASE_URL = (
    "https://raw.githubusercontent.com/lowe-lab-ucl/btrack-examples/main/"
)

POOCH = pooch.create(
    path=pooch.os_cache("btrack-examples"),
    base_url=BASE_URL,
    version_dev="main",
    registry=None,
)

# Get registry file using pooch itself
registry_file = pooch.retrieve(
    url=BASE_URL + "registry.txt",
    known_hash=None,
    fname="registry.txt",
    path=POOCH.path,
)
# Load this registry file
POOCH.load_registry(registry_file)


def example_cell_config() -> os.PathLike:
    file_path = POOCH.fetch("examples/cell_config.json")
    return file_path


def example_segmentation() -> os.PathLike:
    file_path = POOCH.fetch("examples/segmented.tif")
    return file_path


def example_objects() -> os.PathLike:
    file_path = POOCH.fetch("examples/objects.csv")
    return file_path
