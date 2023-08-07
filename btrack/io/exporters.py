from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING, Optional

import numpy as np

from btrack import constants

from .hdf import HDF5FileHandler
from .utils import check_track_type

if TYPE_CHECKING:
    from btrack import BayesianTracker

# get the logger instance
logger = logging.getLogger(__name__)


def export_delegator(
    filename: os.PathLike,
    tracker: BayesianTracker,
    obj_type: Optional[str] = None,
    filter_by: Optional[str] = None,
) -> None:
    """Export data from the tracker using the appropriate exporter.

    Parameters
    ----------
    filename : str
        The filename to export the data. The extension (e.g. .h5) is used
        to select the correct export function.
    tracker : BayesianTracker
        An instance of the tracker.
    obj_type : str, optional
        The object type to export the data. Usually `obj_type_1`
    filter_by : str, optional
        A string that represents how the data has been filtered prior to
        tracking, e.g. using the object property `area>100`

    Notes
    -----
    This uses the appropriate exporter dependent on the given file extension.
    """
    export_dir, export_fn = os.path.split(filename)
    _, ext = os.path.splitext(filename)

    if ext == ".csv":
        export_CSV(filename, tracker.tracks, obj_type=obj_type)
    elif ext in (".hdf", ".hdf5", ".h5"):
        _export_HDF(filename, tracker, obj_type=obj_type, filter_by=filter_by)
    else:
        logger.error(f"Export file format {ext} not recognized.")


def export_CSV(
    filename: os.PathLike,
    tracks: list,
    properties: list = constants.DEFAULT_EXPORT_PROPERTIES,
    obj_type: Optional[str] = None,
):
    """Export the track data as a simple CSV file.

    Parameters
    ----------
    filename : str
        The filename of the file to be exported.
    tracks : list[Tracklet]
        A list of Tracklet objects to be exported.
    properties : list, default = constants.DEFAULT_EXPORT_PROPERTIES
        A list of tracklet properties to be exported.
    obj_type : str, optional
        A string describing the object type, e.g. `obj_type_1`.

    """

    if not tracks:
        logger.error(f"No tracks found when exporting to: {filename}")
        return

    if not check_track_type(tracks):
        logger.error("Tracks of incorrect type")

    logger.info(f"Writing out CSV files to: {filename}")
    export_track = np.vstack([t.to_array(properties) for t in tracks])

    with open(filename, "w", newline="") as csv_file:
        csvwriter = csv.writer(csv_file, delimiter=" ")
        csvwriter.writerow(properties)
        for i in range(export_track.shape[0]):
            csvwriter.writerow(export_track[i, :].tolist())


def export_LBEP(filename: os.PathLike, tracks: list):
    """Export the LBEP table as a text file."""
    if not tracks:
        logger.error(f"No tracks found when exporting to: {filename}")
        return

    if not check_track_type(tracks):
        logger.error("Tracks of incorrect type")

    tracks.sort(key=lambda t: t.ID)

    with open(filename, "w") as lbep_file:
        logger.info(f"Writing LBEP file: {filename}...")
        for track in tracks:
            lbep = f"{track.ID} {track.t[0]} {track.t[-1]} {track.parent}"
            lbep_file.write(f"{lbep}\n")


def _export_HDF(
    filename: os.PathLike,
    tracker,
    obj_type=None,
    filter_by: Optional[str] = None,
):
    """Export to HDF."""

    filename_noext, ext = os.path.splitext(filename)
    if ext != ".h5":
        filename = filename_noext + ".h5"
        logger.warning(f"Changing HDF filename to {filename}")

    with HDF5FileHandler(filename, read_write="a", obj_type=obj_type) as hdf:
        # if there are no objects, write them out
        if f"objects/{obj_type}" not in hdf._hdf:
            hdf.write_objects(tracker)
        # write the tracks
        hdf.write_tracks(tracker, f_expr=filter_by)
