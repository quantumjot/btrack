"""
This module is a reader plugin btrack files for napari.
"""
import os
import pathlib
from collections.abc import Sequence
from typing import Callable, Optional, Union

from napari.types import LayerDataTuple

from btrack.io import HDF5FileHandler
from btrack.utils import tracks_to_napari

# Type definitions
PathOrPaths = Union[os.PathLike, Sequence[os.PathLike]]
ReaderFunction = Callable[[PathOrPaths], list[LayerDataTuple]]


def get_reader(path: PathOrPaths) -> Optional[ReaderFunction]:
    """A basic implementation of the napari_get_reader hook specification.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    supported_extensions = [
        ".h5",
        ".hdf",
        ".hdf5",
    ]
    return (
        reader_function if pathlib.Path(path).suffix in supported_extensions else None
    )


def reader_function(path: PathOrPaths) -> list[LayerDataTuple]:
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "metadata" and "layer_type" are optional. napari will default
        to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = path if isinstance(path, list) else [path]

    # store the layers to be generated
    layers: list[tuple] = []

    for _path in paths:
        with HDF5FileHandler(_path, "r") as hdf:
            # get the segmentation if there is one
            segmentation = hdf.segmentation
            if segmentation is not None:
                layers.append((segmentation, {}, "labels"))

            # iterate over object types and create a layer for each
            for obj_type in hdf.object_types:
                # set the object type, and retrieve the tracks
                hdf.object_type = obj_type

                if f"tracks/{obj_type}" not in hdf._hdf:
                    continue

                tracklets = hdf.tracks
                tracks, properties, graph = tracks_to_napari(tracklets)

                # optional kwargs for the corresponding viewer.add_* method
                # https://napari.org/docs/api/napari.components.html#module-napari.components.add_layers_mixin
                add_kwargs = {
                    "properties": properties,
                    "graph": graph,
                    "name": obj_type,
                    "blending": "translucent",
                }

                layer = (tracks, add_kwargs, "tracks")
                layers.append(layer)
    return layers
