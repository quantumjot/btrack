from __future__ import annotations

import dataclasses
import functools
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy.typing as npt

import numpy as np
from skimage.util import map_array

# import core
from . import _version, btypes, constants
from .btypes import Tracklet
from .constants import DEFAULT_EXPORT_PROPERTIES, Dimensionality
from .io import objects_from_dict
from .io._localization import segmentation_to_objects
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
    if error != constants.Errors.SUCCESS and error != constants.Errors.NO_ERROR:
        logger.error(f"ERROR: {error}")
        return True
    return False


def log_stats(stats: dict) -> None:
    """Take the statistics from the track and log the output."""

    if log_error(stats["error"]):
        return

    logger.info(
        " - Timing (Bayesian updates: {:.2f}ms, Linking:"
        " {:.2f}ms)".format(stats["t_update_belief"], stats["t_update_link"])
    )

    logger.info(
        " - Probabilities (Link: {:.5f}, Lost:"
        " {:.5f})".format(stats["p_link"], stats["p_lost"])
    )

    if stats["complete"]:
        return

    logger.info(
        " - Stats (Active: {:d}, Lost: {:d}, Conflicts "
        "resolved: {:d})".format(
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
        return all(getattr(o, a) >= v[0] and getattr(o, a) <= v[1] for a, v in axes)

    return [o for o in objects if within(o)]


def _lbep_table(tracks: list[btypes.Tracklet]) -> np.array:
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

            if trk_property.ndim > constants.Dimensionality.TWO:
                raise ValueError(
                    f"Track properties of {trk_property.ndim} dimensions are "
                    "not currently supported."
                )

            assert trk_property.shape[0] == len(track)

            if trk_property.ndim == constants.Dimensionality.TWO:
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
    tracks: list[btypes.Tracklet],
    *,
    ndim: int | None = None,
    replace_nan: bool = True,
):
    """Convert a list of Tracklets to napari format input.

    Parameters
    ----------
    tracks : list[btypes.Tracklet]
        A list of tracklet objects from BayesianTracker.
    ndim : int or None
        The number of spatial dimensions of the data. If not specified, the
        function attempts to guess the final dimensionality using the z
        coordinates. If specified, it must have a value of 2 or 3.
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
    # guess the dimensionality from the data by checking whether the non-dummy z values
    # are all zero. If all z are zero then the data are planar, i.e. 2D
    if ndim is None:
        z = np.concatenate(
            [np.asarray(track.z)[~np.asarray(track.dummy)] for track in tracks]
        )
        ndim = Dimensionality.THREE if np.any(z) else Dimensionality.TWO

    if ndim not in (Dimensionality.TWO, Dimensionality.THREE):
        raise ValueError("ndim must be 2 or 3 dimensional.")

    t_header = ["ID", "t"] + ["z", "y", "x"][-ndim:]
    p_header = ["t", "state", "generation", "root", "parent", "dummy"]

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
    def nans_idx(x):
        return x.nonzero()[0]

    if replace_nan:
        for k, v in properties.items():
            nans = np.isnan(v)
            v[nans] = np.interp(nans_idx(nans), nans_idx(~nans), v[~nans])
            properties[k] = v

    graph = {t.ID: [t.parent] for t in ordered if not t.is_root}
    return data, properties, graph


def napari_to_tracks(
    data: npt.NDArray,
    properties: Optional[dict[str, npt.ArrayLike]],
    graph: Optional[dict[int, list[int]]],
) -> list[btypes.Tracklet]:
    """Convert napari Tracks to a list of Tracklets.

    Parameters
    ----------
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

    Returns
    -------
    tracks : list[btypes.Tracklet]
        A list of tracklet objects created from the napari Tracks layer data.

    """

    if data.shape[1] == Dimensionality.FIVE:
        track_id, t, z, y, x = data.T
    elif data.shape[1] == Dimensionality.FOUR:
        track_id, t, y, x = data.T
        z = np.zeros_like(x)
    else:
        raise ValueError(
            "Data must have either 4 (ID, t, y, x) or 5 (ID, t, z, y, x) columns, "
            f"not {data.shape[1]}"
        )

    # Create all PyTrackObjects
    objects_dict = {
        "ID": np.arange(track_id.size),
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "dummy": properties.get("dummy", np.full_like(track_id, fill_value=False)),
        "label": properties.get(
            "state", np.full_like(track_id, fill_value=constants.States.NULL)
        ),
    }
    track_objects = objects_from_dict(objects_dict)

    # Create all Tracklets
    tracklets = []
    for track in np.unique(track_id).astype(int):
        # Create tracklet
        track_indices = np.argwhere(track_id == track).ravel()
        track_data = [track_objects[i] for i in track_indices]
        parent = graph.get(track, [track])[0]
        children = [child for (child, parents) in graph.items() if track in parents]
        tracklet = Tracklet(
            ID=track,
            data=track_data,
            parent=parent,
            children=children,
        )

        # Determine root tracklet
        tracklet.root = parent
        tracklet.generation = 0 if tracklet.root == track else 1
        while tracklet.root in graph:
            tracklet.root = graph[tracklet.root][0]
            tracklet.generation += 1

        tracklets.append(tracklet)

    return tracklets


def update_segmentation(
    segmentation: np.ndarray,
    tracks: list[btypes.Tracklet],
    *,
    scale: Optional[tuple(float)] = None,
    color_by: str = "ID",
) -> np.ndarray:
    """Map tracks back into a masked array.

    Parameters
    ----------
    segmentation : np.array
        Array containing a timeseries of single cell masks. Dimensions should be
        ordered T(Z)YX. Assumes that this is not binary and each object has a unique ID.
    tracks : list[btypes.Tracklet]
        A list of :py:class:`btrack.btypes.Tracklet` objects from BayesianTracker.
    scale : tuple, optional
        A scale for each spatial dimension of the input tracks. Defaults
        to one for all axes, and allows scaling for anisotropic imaging data.
        Dimensions should be ordered XY(Z).
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

    keys.update(
        {
            key: idx
            for idx, key in enumerate(
                tracks[0].properties.keys(), start=max(keys.values()) + 1
            )
        }
    )

    coords_arr = np.concatenate(
        [track.to_array()[~np.array(track.dummy), :].astype(int) for track in tracks]
    )

    scale = tuple([1.0] * (segmentation.ndim - 1)) if scale is None else scale

    if (segmentation.ndim - 1) != len(scale):
        raise ValueError(
            "Scale should have the same number of spatial dimensions as `segmentation`."
        )

    if color_by not in keys:
        raise ValueError(f"Property ``{color_by}`` not found in track.")

    relabeled = np.zeros_like(segmentation)
    for t, single_segmentation in enumerate(segmentation):
        frame_coords = coords_arr[coords_arr[:, 1] == t]

        xc, yc = frame_coords[:, keys["x"]], frame_coords[:, keys["y"]]
        new_id = frame_coords[:, keys[color_by]]

        xc = (xc * scale[0]).astype(int)
        yc = (yc * scale[1]).astype(int)

        if single_segmentation.ndim == constants.Dimensionality.TWO:
            old_id = single_segmentation[yc, xc]
        elif single_segmentation.ndim == constants.Dimensionality.THREE:
            zc = frame_coords[:, keys["z"]]
            zc = (zc * scale[2]).astype(int)
            old_id = single_segmentation[zc, yc, xc]

        relabeled[t] = map_array(single_segmentation, old_id, new_id) * (
            single_segmentation > 0
        )

    return relabeled


@dataclasses.dataclass(frozen=True, init=False)
class SystemInformation:
    btrack_version: str = _version.version
    system_platform: str = constants.BTRACK_PLATFORM
    system_python: str = constants.BTRACK_PYTHON_VERSION

    def __repr__(self) -> str:
        # override to have slightly nicer formatting
        return "\n".join(
            [f"{key}: {value}" for key, value in dataclasses.asdict(self).items()]
        )


def log_debug_info(fn):
    """Wrapper to provide additional debug info when loading a shared library
    or any other function that needs special debugging info."""

    @functools.wraps(fn)
    def wrapped_func_to_debug(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as err:
            debug_info = dataclasses.asdict(SystemInformation())
            exception_info = {
                "function": fn,
                "exception": err,
                "arguments": args,
                "keywords": kwargs,
            }

            debug_info.update(exception_info)
            debug_str = "\n".join(
                [f" - {key}: {value}" for key, value in debug_info.items()]
            )

            logger.error(f"Exception caught:\n{debug_str}")

            raise Exception from err

    return wrapped_func_to_debug
