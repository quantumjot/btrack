from __future__ import annotations

import logging
from typing import Any, Union

import numpy as np

# import core
from btrack import btypes, constants

# get the logger instance
logger = logging.getLogger(__name__)


def localizations_to_objects(
    localizations: Union[
        np.ndarray, list[btypes.PyTrackObject], dict[str, Any]
    ]
) -> list[btypes.PyTrackObject]:
    """Take a numpy array or pandas dataframe and convert to PyTrackObjects.

    Parameters
    ----------
    localizations : list[PyTrackObject], np.ndarray, pandas.DataFrame
        A list or array of localizations.

    Returns
    -------
    objects : list[PyTrackObject]
        A list of PyTrackObject objects that represent the localizations.
    """

    logger.info(f"Objects are of type: {type(localizations)}")

    if isinstance(localizations, list) and check_object_type(localizations):
        # if these are already PyTrackObjects just silently return
        return localizations

    # do we have a numpy array or pandas dataframe?
    if isinstance(localizations, np.ndarray):
        return objects_from_array(localizations)
    else:
        try:
            objects_dict = {
                c: np.asarray(localizations[c]) for c in localizations
            }
        except ValueError as err:
            logger.error(f"Unknown localization type: {type(localizations)}")
            raise TypeError(
                f"Unknown localization type: {type(localizations)}"
            ) from err

    # how many objects are there
    n_objects = objects_dict["t"].shape[0]
    objects_dict["ID"] = np.arange(n_objects)

    return objects_from_dict(objects_dict)


def objects_from_dict(objects_dict: dict) -> list[btypes.PyTrackObject]:
    """Construct PyTrackObjects from a dictionary"""
    # now that we have the object dictionary, convert this to objects
    objects = []
    n_objects = int(objects_dict["t"].shape[0])

    assert all(v.shape[0] == n_objects for k, v in objects_dict.items())

    for i in range(n_objects):
        data = {k: v[i] for k, v in objects_dict.items()}
        obj = btypes.PyTrackObject.from_dict(data)
        objects.append(obj)
    return objects


def objects_from_array(
    objects_arr: np.ndarray,
    *,
    default_keys: list[str] = constants.DEFAULT_OBJECT_KEYS,
) -> list[btypes.PyTrackObject]:
    """Construct PyTrackObjects from a numpy array."""
    assert objects_arr.ndim == constants.Dimensionality.TWO

    MIN_N_FEATURES = 3
    n_features = objects_arr.shape[1]
    assert n_features >= MIN_N_FEATURES

    n_objects = objects_arr.shape[0]

    keys = default_keys[:n_features]
    objects_dict = {keys[i]: objects_arr[:, i] for i in range(n_features)}
    objects_dict["ID"] = np.arange(n_objects)
    return objects_from_dict(objects_dict)


def check_track_type(tracks: list) -> bool:
    return all(isinstance(t, btypes.Tracklet) for t in tracks)


def check_object_type(objects: list) -> bool:
    return all(isinstance(o, btypes.PyTrackObject) for o in objects)
