import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, validator

from . import constants
from .btypes import ImagingVolume
from .models import HypothesisModel, MotionModel, ObjectModel
from .utils import read_hypothesis_model, read_motion_model, read_object_model

# get the logger instance
logger = logging.getLogger(__name__)


class TrackerConfig(BaseModel):
    """Configuration for `BayesianTracker`.

    Parameters
    ----------
    name : str
        A name identifier for the model.
    version : str
        A string representing the version of `btrack` used.
    verbose : bool
        A flag to set the verbosity level while logging the output.
    motion_model : Optional[MotionModel]
        The `btrack` motion model. See `models.MotionModel` for more details.
    object_model : Optional[ObjectModel]
        The `btrack` object model. See `models.ObjectModel` for more details.
    hypothesis_model : Optional[HypothesisModel]
        The `btrack` hypothesis model. See `models.HypothesisModel` for more
        details.
    max_search_radius : float
        The maximum search radius of the algorithm in isotropic units of the
        data. Should be greater than zero.
    return_kalman : bool
        Flag to request the Kalman debug info when returning tracks.
    volume : Optional[ImagingVolume]
        The imaging volume as [(xlo, xhi), ..., (zlo, zhi)]. See
        `btypes.ImagingVolume` for more details.
    update_method : constants.BayesianUpdates
        The method to perform the bayesian updates during tracklet linking.
            BayesianUpdates.EXACT
                Use the exact Bayesian update method. Can be slow for systems
                with many objects.
            BayesianUpdates.APPROXIMATE
                Use the approximate Bayesian update method. Useful for systems
                with may objects.
            BayesianUpdates.CUDA
                Use the CUDA implementation of the Bayesian update method. Not
                currently implemented.
    optimizer_options: dict
        Additional options to pass to the optimizer. See `cvxopt.glpk` for more
        details of options that can be passed.

    Notes
    -----
    TODO(arl): add more validation to parameters.
    """

    name: str = "Default"
    version: str = constants.get_version()
    verbose: bool = False
    motion_model: Optional[MotionModel] = None
    object_model: Optional[ObjectModel] = None
    hypothesis_model: Optional[HypothesisModel] = None
    max_search_radius: float = constants.MAX_SEARCH_RADIUS
    return_kalman: bool = False
    volume: Optional[ImagingVolume] = None
    update_method: constants.BayesianUpdates = constants.BayesianUpdates.EXACT
    optimizer_options: dict = constants.GLPK_OPTIONS

    @validator("volume", pre=True, always=True)
    def parse_volume(cls, v):
        if isinstance(v, tuple):
            return ImagingVolume(*v)
        return v

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        json_encoders = {
            np.ndarray: lambda x: x.ravel().tolist(),
        }


def load_config(filename: os.PathLike) -> TrackerConfig:
    """Load a tracker configuration from a file.

    Parameters
    ----------
    filename : os.PathLike
        The filename to load the file.

    Returns
    -------
    cfg : TrackerConfig
        The tracker configuration.
    """
    logger.info(f"Loading configuration file: {filename}")
    filename = Path(filename)

    with open(filename, "r") as json_file:
        json_data = json.load(json_file)

    try:
        cfg = _load_legacy_config(json_data)
    except KeyError:
        cfg = _load_config(json_data)

    assert cfg.motion_model is not None
    return cfg


def _load_legacy_config(json_data: dict) -> TrackerConfig:
    """Load a legacy config file."""
    config = json_data["TrackerConfig"]

    t_config = {
        "motion_model": read_motion_model(config),
        "object_model": read_object_model(config),
        "hypothesis_model": read_hypothesis_model(config),
    }

    return TrackerConfig(**t_config)


def _load_config(json_data: dict) -> TrackerConfig:
    """Load a new style config from a JSON file."""
    return TrackerConfig(**json_data)


def save_config(filename: os.PathLike, cfg: TrackerConfig) -> None:
    """Save the config to a JSON file.

    Parameters
    ----------
    filename : os.PathLike
        The filename to save the configuration file.
    cfg : TrackerConfig
        The tracker configuration to save.
    """

    with open(filename, "w") as json_file:
        json_data = json.loads(cfg.json())
        json.dump(json_data, json_file, indent=2, separators=(",", ": "))
