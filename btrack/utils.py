from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from skimage.util import map_array

# import core
from . import btypes, constants
from ._localization import segmentation_to_objects
from .constants import Dimensionality
from .models import HypothesisModel, MotionModel, ObjectModel

# Choose a subset of classes/functions to document in public facing API
__all__ = [
    "segmentation_to_objects",
    "update_segmentation",
]

# get the logger instance
logger = logging.getLogger(__name__)


def log_error(err_code) -> bool:
    """Take an error code from the tracker and log an error for the user."""
    error = constants.Errors(err_code)
    if (
        error != constants.Errors.SUCCESS
        and error != constants.Errors.NO_ERROR
    ):
        logger.error(f"ERROR: {error}")
        return True
    return False


def log_stats(stats: dict) -> None:
    """Take the statistics from the track and log the output."""

    if log_error(stats["error"]):
        return

    logger.info(
        " - Timing (Bayesian updates: {0:.2f}ms, Linking:"
        " {1:.2f}ms)".format(stats["t_update_belief"], stats["t_update_link"])
    )

    logger.info(
        " - Probabilities (Link: {0:.5f}, Lost:"
        " {1:.5f})".format(stats["p_link"], stats["p_lost"])
    )

    if stats["complete"]:
        return

    logger.info(
        " - Stats (Active: {0:d}, Lost: {1:d}, Conflicts "
        "resolved: {2:d})".format(
            stats["n_active"], stats["n_lost"], stats["n_conflicts"]
        )
    )


def read_motion_model(cfg: dict) -> Optional[MotionModel]:
    cfg = cfg.get("MotionModel", {})
    if not cfg:
        return None
    return MotionModel(**cfg)


def read_object_model(cfg: dict) -> Optional[ObjectModel]:
    cfg = cfg.get("ObjectModel", {})
    if not cfg:
        return None
    return ObjectModel(**cfg)


def read_hypothesis_model(cfg: dict) -> Optional[HypothesisModel]:
    cfg = cfg.get("HypothesisModel", {})
    if not cfg:
        return None
    return HypothesisModel(**cfg)


def crop_volume(objects, volume=constants.VOLUME):
    """Return a list of objects that fall within a certain volume."""
    axes = zip(["x", "y", "z", "t"], volume)

    def within(o):
        return all(
            [getattr(o, a) >= v[0] and getattr(o, a) <= v[1] for a, v in axes]
        )

    return [o for o in objects if within(o)]


def _cat_tracks_as_dict(
    tracks: list[btypes.Tracklet], properties: list
) -> dict:
    """Concatenate all tracks a dictionary."""
    assert all([isinstance(t, btypes.Tracklet) for t in tracks])

    data: dict = {}

    for track in tracks:
        trk = track.to_dict(properties)

        if not data:
            data = {k: [] for k in trk.keys()}

        for key in data.keys():
            property = trk[key]
            if not isinstance(property, (list, np.ndarray)):
                property = [property] * len(track)

            assert len(property) == len(track)
            data[key].append(property)

    for key in data.keys():
        data[key] = np.concatenate(data[key])

    return data


def tracks_to_napari(
    tracks: list[btypes.Tracklet],
    ndim: int = 3,
    coords_axes: Optional[list] = None,
    replace_nan: bool = True,
):
    """Convert a list of Tracklets to napari format input.

    Parameters
    ----------
    tracks : list[btypes.Tracklet]
        A list of tracklet objects from BayesianTracker.
    ndim : int
        The number of spatial dimensions of the data. Must be 2 or 3.
    coords_axes: list
        The order of axes in the track objects. Defaults to ["z", "y", "x"].
        For older files, use ["z", "x", "y"] to sync with segmentation mask.
    replace_nan : bool
        Replace instances of NaN/inf in the track properties with an
        interpolated value.

    Returns
    -------
    data : array (N, D+1)
        Coordinates for N points in D+1 dimensions. ID,T,(Z),Y,X. The first
        axis is the integer ID of the track. D is either 3 or 4 for planar
        or volumetric timeseries respectively.
    properties : dict {str: array (N,)}
        Properties for each point. Each property should be an array of length N,
        where N is the number of points.
    graph : dict {int: list}
        Graph representing associations between tracks. Dictionary defines the
        mapping between a track ID and the parents of the track. This can be
        one (the track has one parent, and the parent has >=1 child) in the
        case of track splitting, or more than one (the track has multiple
        parents, but only one child) in the case of track merging.
    """
    # TODO: arl guess the dimensionality from the data
    if ndim not in (Dimensionality.TWO, Dimensionality.THREE):
        raise ValueError("ndim must be 2 or 3 dimensional.")

    coords_axes = ["z", "y", "x"] if coords_axes is None else coords_axes
    assert isinstance(coords_axes, list)

    t_header = ["ID", "t"] + coords_axes[-ndim:]
    p_header = ["t", "state", "generation", "root", "parent"]

    # ensure lexicographic ordering of tracks
    ordered = sorted(list(tracks), key=lambda t: t.ID)
    header = t_header + p_header
    tracks_as_dict = _cat_tracks_as_dict(ordered, header)

    # note that there may be other metadata in the tracks, grab that too
    prop_keys = p_header + [
        k for k in tracks_as_dict.keys() if k not in t_header
    ]

    # get the data for napari
    data = np.stack(
        [v for k, v in tracks_as_dict.items() if k in t_header], axis=1
    )
    properties = {k: v for k, v in tracks_as_dict.items() if k in prop_keys}

    # replace any NaNs in the properties with an interpolated value
    if replace_nan:
        for k, v in properties.items():
            nans = np.isnan(v)
            nans_idx = lambda x: x.nonzero()[0]
            v[nans] = np.interp(nans_idx(nans), nans_idx(~nans), v[~nans])
            properties[k] = v

    graph = {t.ID: [t.parent] for t in ordered if not t.is_root}
    return data, properties, graph


def update_segmentation(
    segmentation: np.ndarray, tracks: list[btypes.Tracklet]
) -> np.ndarray:
    """
    Map btrack output tracks back into a masked array.

    Parameters
    ----------
    segmentation : np.array
        Array containing a timeseries of single cell masks. Dimensions should be
        ordered T(Z)YX. Assumes that this is not binary and each object has a unique ID.
    tracks : list[btypes.Tracklet]
        A list of :py:class:`btrack.btypes.Tracklet` objects from BayesianTracker.

    Returns
    -------
    relabeled : np.array
        Array containing the same masks as segmentation but relabeled to
        maintain single cell identity over time.

    Example
    -------

    import btrack
    tracker = btrack.BayesianTracker()
    objects = btrack.utils.segmentation_to_objects(segmentation)
    tracker.append(objects)
    ...
    tracker.optimize()
    tracks = tracker.tracks

    tracked_segmentation = btrack.utils.update_segmentation(
                                    segmentation, tracks)
    """

    coords_arr = np.concatenate(
        [
            track.to_array()[~np.array(track.dummy)][:, :5].astype(int)
            for track in tracks
        ]
    )
    relabeled = np.zeros_like(segmentation)
    for t, single_segmentation in enumerate(segmentation):
        frame_coords = coords_arr[coords_arr[:, 1] == t]
        new_id, tc, xc, yc, zc = tuple(frame_coords.T)
        if single_segmentation.ndim == 2:
            old_id = single_segmentation[yc, xc]
        elif single_segmentation.ndim == 3:
            old_id = single_segmentation[zc, yc, xc]

        relabeled[t] = map_array(single_segmentation, old_id, new_id) * (
            single_segmentation > 0
        )

    return relabeled


if __name__ == "__main__":
    pass
