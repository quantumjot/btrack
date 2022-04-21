"""
This module is a reader plugin btrack files for napari.
"""
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from napari.types import LayerDataTuple
from napari_plugin_engine import napari_hook_implementation

from btrack.dataio import HDF5FileHandler
from btrack.utils import tracks_to_napari

# Type definitions
PathLike = Union[str, Path]
PathOrPaths = Union[PathLike, Sequence[PathLike]]
ReaderFunction = Callable[[PathOrPaths], List[LayerDataTuple]]


@napari_hook_implementation
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
    return reader_function


def reader_function(path: PathLike) -> List[LayerDataTuple]:
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
        layer. Both "meta", and "layer_type" are optional. napari will default
        to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if not isinstance(path, list) else path

    # store the layers to be generated
    layers = []

    for _path in paths:
        with HDF5FileHandler(_path, "r") as hdf:

            # get the segmentation if there is one
            import pdb; pdb.set_trace()
            if "segmentation" in hdf._hdf:
                segmentation = hdf.segmentation
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
