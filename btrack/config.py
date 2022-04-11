import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

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
    version : str
    verbose : bool
    motion_model : Optional[MotionModel]
    object_model : Optional[ObjectModel]
    hypothesis_model : Optional[HypothesisModel]
    max_search_radius : float
    return_kalman : bool
    volume : Optional[ImagingVolume]
    update_method : constants.BayesianUpdates
    optimizer_options: dict
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

    class Config:
        arbitrary_types_allowed = True
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
