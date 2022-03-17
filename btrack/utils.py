#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Name:     BayesianTracker
# Purpose:  A multi object tracking library, specifically used to reconstruct
#           tracks in crowded fields. Here we use a probabilistic network of
#           information to perform the trajectory linking. This method uses
#           positional and visual information for track linking.
#
# Authors:  Alan R. Lowe (arl) a.lowe@ucl.ac.uk
#
# License:  See LICENSE.md
#
# Created:  14/08/2014
# ------------------------------------------------------------------------------


__author__ = "Alan R. Lowe"
__email__ = "code@arlowe.co.uk"


import dataclasses
import json
import logging
import os

import numpy as np

# import core
from . import btypes, constants
from ._localization import segmentation_to_objects
from .models import HypothesisModel, MotionModel, ObjectModel

# get the logger instance
logger = logging.getLogger(__name__)


# add an alias here
segmentation_to_objects = segmentation_to_objects


def load_config(filename: os.PathLike) -> dict:
    """Load a tracking configuration file."""
    if not os.path.exists(filename):
        # check whether it exists in the user model directory
        _, fn = os.path.split(filename)
        local_filename = os.path.join(constants.USER_MODEL_DIR, fn)

        if not os.path.exists(local_filename):
            logger.error(f"Configuration file {filename} not found")
            raise IOError(f"Configuration file {filename} not found")
        else:
            filename = local_filename

    with open(filename, "r") as config_file:
        config = json.load(config_file)

    if "TrackerConfig" not in config:
        logger.error("Configuration file is malformed.")
        raise Exception("Tracking config is malformed")

    config = config["TrackerConfig"]

    logger.info(f"Loading configuration file: {filename}")
    t_config = {
        "MotionModel": read_motion_model(config),
        "ObjectModel": read_object_model(config),
        "HypothesisModel": read_hypothesis_model(config),
    }

    return t_config


def log_error(err_code):
    """Take an error code from the tracker and log an error for the user."""
    error = constants.Errors(err_code)
    if (
        error != constants.Errors.SUCCESS
        and error != constants.Errors.NO_ERROR
    ):
        logger.error(f"ERROR: {error}")
        return True
    return False


def log_stats(stats):
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


def read_motion_model(config: dict) -> MotionModel:
    """Read a motion model from a configuration dictionary.

    Read in a motion model description file and return a dictionary containing
    the appropriate parameters. See `models.MotionModel` for more details of the
    parameters.

    Parameters
    ----------
    config : dict
        A dictionary describing the motion model.

    Returns
    -------
    model : MotionModel
        A `models.MotionModel` instance to configure BayesianTracker.

    Notes
    -----
    Motion models can be described using JSON format, with a basic structure
    as follows:

        {
          "MotionModel":{
            "name": "ConstantVelocity",
            "dt": 1.0,
            "measurements": 3,
            "states": 6,
            "accuracy": 2.0,
            "A": {
              "matrix": [1,0,0,1,0,0,...
              ...
              ] }
            }
        }

    Matrices are flattened JSON arrays.

    Most are self explanatory, except accuracy (perhaps a misnoma) - this
    represents the integration limits when determining the probabilities from
    the multivariate normal distribution.

    Note that the matrices are stored as 1D matrices here. In the future,
    this could form part of a Python only motion model.

    TODO(arl): More parsing of the data/reshaping arrays. Raise an
    appropriate error if there is something wrong with the model
    definition.
    """

    if "MotionModel" not in list(config.keys()):
        raise ValueError("Not a valid motion model in configuration.")

    motion_config = config["MotionModel"]
    if not motion_config:
        return None

    matrices = frozenset(["A", "H", "P", "G", "R"])
    fields = [f.name for f in dataclasses.fields(MotionModel)]

    model_kwargs = {}

    for field in fields:
        if field not in motion_config.keys():
            logger.error(f"Key {field} not found in `MotionModel` config.")

        # if this is a matrix, prepare it
        if field in matrices:
            if "sigma" in motion_config[field]:
                sigma = motion_config[field]["sigma"]
            else:
                sigma = 1.0
            matrix = np.array(motion_config[field]["matrix"], dtype=np.float64)
            model_kwargs[field] = matrix * sigma
        else:
            model_kwargs[field] = motion_config[field]

    # set some standard params
    model = MotionModel(**model_kwargs)

    # call the reshape function to set the matrices to the correct shapes
    model.reshape()
    return model


def read_object_model(config: dict) -> ObjectModel:
    """Read an object model from a configuration dictionary.

    Read in a object model description file and return a dictionary containing
    the appropriate parameters. See `models.ObjectModel` for more details of the
    parameters.

    Parameters
    ----------
    config : dict
        A dictionary describing the object model.

    Returns
    -------
    model : ObjectModel
        A `models.ObjectModel` instance to configure BayesianTracker.

    Notes
    -----
    Object models can be described using JSON format, with a basic structure
    as follows:

        {
          "ObjectModel":{
            "name": "UniformState",
            "states": 1,
            "transition": {
              "matrix": [1] }
              ...
            }
        }

    Matrices are flattened JSON arrays.

    Note that the matrices are stored as 1D matrices here. In the future,
    this could form part of a Python only object model.

    TODO(arl): More parsing of the data/reshaping arrays. Raise an
    appropriate error if there is something wrong with the model definition
    """
    matrices = frozenset(["transition", "emission", "start"])

    if "ObjectModel" not in list(config.keys()):
        raise ValueError("Not a valid object model file")

    object_config = config["ObjectModel"]
    if not object_config:
        return None

    model = ObjectModel()

    # set some standard params
    model.name = object_config["name"].encode("utf-8")
    model.states = object_config["states"]

    for matrix in matrices:
        m_data = np.array(object_config[matrix]["matrix"], dtype="float")
        setattr(model, matrix, m_data)

    # call the reshape function to set the matrices to the correct shapes
    model.reshape()
    return model


def read_hypothesis_model(config: dict) -> HypothesisModel:
    """Read a hypothesis model from a configuration dictionary.

    Read in a hypothesis model description file and return a dictionary
    containing the appropriate parameters. See `models.ObjectModel` for more
    details of the parameters.

    Parameters
    ----------
    config : dict
        A dictionary describing the object model.

    Returns
    -------
    model : HypothesisModel
        A `models.HypothesistModel` instance to configure BayesianTracker.

    Read in a set of hypothesis parameters from a JSON description file.  The
    JSON file should contain the parameters of the PyHypothesisParams structure
    and the function will return an instantiated PyHypothesisParams to be
    passed to the optimisation engine.

    Args:
        filename: the filename of the parameter file

    Notes:
        None
    """
    if "ObjectModel" not in list(config.keys()):
        raise ValueError("Not a valid object model file")

    hypothesis_config = config["HypothesisModel"]
    if not hypothesis_config:
        return None

    fields = [f.name for f in dataclasses.fields(HypothesisModel)]

    for field in fields:
        if field not in hypothesis_config.keys():
            logger.error(f"Key {field} not found in `HypothesisModel` config.")

    model = HypothesisModel(**hypothesis_config)

    return model


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
