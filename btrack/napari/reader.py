"""
This module is a reader plugin btrack files for napari.
"""
from __future__ import annotations

import os
from typing import Union, List, Callable, Sequence

from napari.types import LayerDataTuple
from napari_plugin_engine import napari_hook_implementation

from btrack.io import HDF5FileHandler
from btrack.utils import tracks_to_napari

# Type definitions
PathOrPaths = Union[os.PathLike, Sequence[os.PathLike]]
ReaderFunction = Callable[[PathOrPaths], List[LayerDataTuple]]  # noqa: UP006


@napari_hook_implementation
def get_reader(path: PathOrPaths) -> ReaderFunction | None:
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
    return reader_function


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
                tracks, properties, graph = tracks_to_napari(tracklets, ndim=2)

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
