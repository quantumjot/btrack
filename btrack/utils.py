from __future__ import annotations

import logging

import numpy as np
from numpy import typing as npt
from skimage.util import map_array

# import core
from btrack import btypes, constants
from btrack.constants import DEFAULT_EXPORT_PROPERTIES, Dimensionality
from btrack.io._localization import segmentation_to_objects
from btrack.models import HypothesisModel, MotionModel, ObjectModel

# Choose a subset of classes/functions to document in public facing API
__all__ = [
    "segmentation_to_objects",
    "update_segmentation",
]

# get the logger instance
logger = logging.getLogger(__name__)

TWO_DIM = 2
THREE_DIM = 3


def log_error(err_code) -> bool:
    """Take an error code from the tracker and log an error for the user."""
    error = constants.Errors(err_code)
    if error not in [constants.Errors.SUCCESS, constants.Errors.NO_ERROR]:
        logger.error(f"ERROR: {error}")
        return True
    return False


def log_stats(stats: dict) -> None:
    """Take the statistics from the track and log the output."""

    if log_error(stats["error"]):
        return

    logger.info(
        f" - Timing (Bayesian updates: {stats['t_update_belief']:.2f}"
        f"ms, Linking: {stats['t_update_link']:.2f}ms)"
    )

    logger.info(
        f" - Probabilities (Link: {stats['p_link']:.5f}, "
        f"Lost: {stats['p_lost']:.5f})"
    )

    if stats["complete"]:
        return

    logger.info(
        f" - Stats (Active: {stats['n_active']:d}, Lost: {stats['n_lost']:d}, "
        f"Conflicts resolved: {stats['n_conflicts']:d})"
    )


def read_motion_model(cfg: dict) -> MotionModel | None:
    cfg = cfg.get("MotionModel", {})
    return MotionModel(**cfg) if cfg else None


def read_object_model(cfg: dict) -> ObjectModel | None:
    cfg = cfg.get("ObjectModel", {})
    return ObjectModel(**cfg) if cfg else None


def read_hypothesis_model(cfg: dict) -> HypothesisModel | None:
    cfg = cfg.get("HypothesisModel", {})
    return HypothesisModel(**cfg) if cfg else None


def crop_volume(objects, volume=constants.VOLUME):
    """Return a list of objects that fall within a certain volume."""
    axes = zip(["x", "y", "z", "t"], volume)

    def within(o):
        return all(getattr(o, a) >= v[0] and getattr(o, a) <= v[1] for a, v in axes)

    return [o for o in objects if within(o)]


def _lbep_table(tracks: list[btypes.Tracklet]) -> npt.NDArray:
    """Create an LBEP table from a track."""
    return np.asarray([trk.LBEP() for trk in tracks], dtype=np.int32)


def _cat_tracks_as_dict(tracks: list[btypes.Tracklet], properties: list[str]) -> dict:
    """Concatenate all tracks as dictionary."""
    assert all(isinstance(t, btypes.Tracklet) for t in tracks)

    data: dict = {}

    for track in tracks:
        trk = track.to_dict(properties)

        for key in trk:
            trk_property = np.asarray(trk[key])

            # if we have a scalar value, repeat it so the dimensions match
            if trk_property.ndim == 0:
                trk_property = np.repeat(trk_property, len(track))

            if trk_property.ndim > TWO_DIM:
                raise ValueError(
                    f"Track properties of {trk_property.ndim} dimensions are "
                    "not currently supported."
                )

            assert trk_property.shape[0] == len(track)

            if trk_property.ndim == TWO_DIM:
                for idx in range(trk_property.shape[-1]):
                    tmp_key = f"{key}-{idx}"
                    if tmp_key not in data:
                        data[tmp_key] = []
                    data[tmp_key].append(trk_property[..., idx])

            else:
                if key not in data:
                    data[key] = []
                data[key].append(trk_property)

    for key in data:
        data[key] = np.concatenate(data[key])

    return data


def tracks_to_napari(
    tracks: list[btypes.Tracklet], ndim: int = 3, *, replace_nan: bool = True
):
    """Convert a list of Tracklets to napari format input.

    Parameters
    ----------
    tracks : list[btypes.Tracklet]
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

    Notes
    -----
    Track properties that are multi-dimensional (>1 dim) will be split according
    to dimension and returned as separate keys. For example, property `softmax`,
    with dimensions (5,) would be split into `softmax-0` ... `softmax-4` for
    representation in napari.
    """
    # TODO: arl guess the dimensionality from the data
    if ndim not in (Dimensionality.TWO, Dimensionality.THREE):
        raise ValueError("ndim must be 2 or 3 dimensional.")

    t_header = ["ID", "t"] + ["z", "y", "x"][-ndim:]
    p_header = ["t", "state", "generation", "root", "parent"]

    # ensure lexicographic ordering of tracks
    ordered = sorted(tracks, key=lambda t: t.ID)
    header = t_header + p_header
    tracks_as_dict = _cat_tracks_as_dict(ordered, header)

    # note that there may be other metadata in the tracks, grab that too
    prop_keys = p_header + [k for k in tracks_as_dict if k not in t_header]

    # get the data for napari
    data = np.stack([v for k, v in tracks_as_dict.items() if k in t_header], axis=1)
    properties = {k: v for k, v in tracks_as_dict.items() if k in prop_keys}

    # replace any NaNs in the properties with an interpolated value
    if replace_nan:
        for k, v in properties.items():
            nans = np.isnan(v)

            def nans_idx(x):
                return x.nonzero()[0]

            v[nans] = np.interp(nans_idx(nans), nans_idx(~nans), v[~nans])
            properties[k] = v

    graph = {t.ID: [t.parent] for t in ordered if not t.is_root}
    return data, properties, graph


def update_segmentation(
    segmentation: npt.NDArray,
    tracks: list[btypes.Tracklet],
    *,
    color_by: str = "ID",
) -> npt.NDArray:
    """Map tracks back into a masked array.

    Parameters
    ----------
    segmentation : np.array
        Array containing a timeseries of single cell masks. Dimensions should be
        ordered T(Z)YX. Assumes that this is not binary and each object has a unique ID.
    tracks : list[btypes.Tracklet]
        A list of :py:class:`btrack.btypes.Tracklet` objects from BayesianTracker.
    color_by : str, default = "ID"
        A value to recolor the segmentation by.

    Returns
    -------
    relabeled : np.array
        Array containing the same masks as segmentation but relabeled to
        maintain single cell identity over time.

    Notes
    -----
    Useful for recoloring a segmentation by a property such as track ID or
    root tree node. Currently the property must be an integer value, greater
    than zero.

    Example
    -------
    >>> tracked_segmentation = btrack.utils.update_segmentation(
    ...     segmentation, tracks, color_by="ID",
    ... )
    """

    keys = {k: i for i, k in enumerate(DEFAULT_EXPORT_PROPERTIES)}

    coords_arr = np.concatenate(
        [
            track.to_array()[~np.array(track.dummy), : len(keys)].astype(int)
            for track in tracks
        ]
    )

    if color_by not in keys:
        raise ValueError(f"Property ``{color_by}`` not found in track.")

    relabeled = np.zeros_like(segmentation)
    for t, single_segmentation in enumerate(segmentation):
        frame_coords = coords_arr[coords_arr[:, 1] == t]

        xc, yc = frame_coords[:, keys["x"]], frame_coords[:, keys["y"]]
        new_id = frame_coords[:, keys[color_by]]

        if single_segmentation.ndim == TWO_DIM:
            old_id = single_segmentation[yc, xc]
        elif single_segmentation.ndim == THREE_DIM:
            old_id = single_segmentation[frame_coords[:, keys["z"]], yc, xc]

        relabeled[t] = map_array(single_segmentation, old_id, new_id) * (
            single_segmentation > 0
        )

    return relabeled
