"""
This module is a writer plugin to export Tracks layers using BTrack
"""

from typing import Optional

import numpy.typing as npt

from btrack.io import HDF5FileHandler
from btrack.utils import napari_to_tracks


def export_to_hdf(
    path: str,
    data: npt.ArrayLike,
    meta: dict,
) -> Optional[str]:
    tracks = napari_to_tracks(
        data=data,
        properties=meta.get("properties", {}),
        graph=meta.get("graph", {}),
    )

    with HDF5FileHandler(
        filename=path,
        read_write="w",
        obj_type="obj_type_1",
    ) as writer:
        writer.write_tracks(tracks)

    return path
