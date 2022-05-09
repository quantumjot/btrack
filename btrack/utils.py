import logging

import numpy as np

# import core
from . import btypes, constants
from ._localization import segmentation_to_objects
from .models import HypothesisModel, MotionModel, ObjectModel

# Choose a subset of classes/functions to document in public facing API
__all__ = ["segmentation_to_objects"]

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


def read_motion_model(cfg: dict) -> MotionModel:
    cfg = cfg.get("MotionModel", {})
    if not cfg:
        return None
    return MotionModel(**cfg)


def read_object_model(cfg: dict) -> ObjectModel:
    cfg = cfg.get("ObjectModel", {})
    if not cfg:
        return None
    return ObjectModel(**cfg)


def read_hypothesis_model(cfg: dict) -> HypothesisModel:
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


def _cat_tracks_as_dict(tracks: list, properties: list):
    """Concatenate all tracks a dictionary."""
    assert all([isinstance(t, btypes.Tracklet) for t in tracks])

    data = {}

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


def tracks_to_napari(tracks: list, ndim: int = 3, replace_nan: bool = True):
    """Convert a list of Tracklets to napari format input.

    Parameters
    ----------
    tracks : list
        A list of tracklet objects from BayesianTracker.
    ndim : int
        The number of spatial dimensions of the data. Must be 2 or 3.
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
    assert ndim in (2, 3)
    t_header = ["ID", "t"] + ["z", "y", "x"][-ndim:]
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


def _pandas_html_repr(obj):
    """Prepare data for HTML representation in a notebook."""
    try:
        import pandas as pd
    except ImportError:
        return (
            "<b>Install pandas for nicer, tabular rendering.</b> <br>"
            + obj.__repr__()
        )

    obj_as_dict = obj.to_dict()

    # now try to process for display in the notebook
    if hasattr(obj, "__len__"):
        n_items = len(obj)
    else:
        n_items = 1

    for k, v in obj_as_dict.items():
        if not isinstance(v, (list, np.ndarray)):
            obj_as_dict[k] = [v] * n_items
        elif isinstance(v, np.ndarray):
            ndim = 0 if n_items == 1 else 1
            if v.ndim > ndim:
                obj_as_dict[k] = [f"{v.shape[ndim:]} array"] * n_items

    return pd.DataFrame.from_dict(obj_as_dict).to_html()


if __name__ == "__main__":
    pass
