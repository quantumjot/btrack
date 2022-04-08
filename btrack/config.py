import json
import logging
import os
from typing import NamedTuple, Optional, Tuple

from pydantic import BaseModel, validator

from . import constants
from .models import HypothesisModel, MotionModel, ObjectModel
from .utils import read_hypothesis_model, read_motion_model, read_object_model

# get the logger instance
logger = logging.getLogger(__name__)


class ImagingVolume(NamedTuple):
    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]


class TrackerConfig(BaseModel):
    """Configuration for `BayesianTracker`.

    Parameters
    ----------
    name : str
    verbose : bool
    motion_model : Optional[MotionModel]
    object_model : Optional[ObjectModel]
    hypothesis_model : Optional[HypothesisModel]
    max_search_radius : float = constants.MAX_SEARCH_RADIUS
    return_kalman : bool = False
    frame_range : Tuple[int] = (0, 0)
    volume : Tuple[Tuple[float, float], Tuple[float], Tuple[float]] = ((0, 0), (0, 0), (0, 0))
    update_method : constants.BayesianUpdates
    optimizer_options: dict

    """

    name: str = "Default"
    verbose: bool = False
    motion_model: Optional[MotionModel] = None
    object_model: Optional[ObjectModel] = None
    hypothesis_model: Optional[HypothesisModel] = None
    max_search_radius: float = constants.MAX_SEARCH_RADIUS
    return_kalman: bool = False
    frame_range: Tuple[int] = (0, 0)
    volume: Optional[ImagingVolume] = None
    update_method: constants.BayesianUpdates = constants.BayesianUpdates.EXACT
    optimizer_options: dict = constants.GLPK_OPTIONS

    @validator("name")
    def testit(cls, n):
        if n != "Alan":
            raise Exception
        return n

    class Config:
        arbitrary_types_allowed = True


PATH = "/Users/arl/Dropbox/Code/py3/BayesianTracker/models/cell_config.json"


def load_config(filename: os.PathLike) -> TrackerConfig:
    pass


def _load_config_legacy(filename: os.PathLike = PATH) -> TrackerConfig:
    """Load a legacy config file."""
    with open(filename, "r") as config_file:
        config = json.load(config_file)

    config = config["TrackerConfig"]

    logger.info(f"Loading configuration file: {filename}")
    t_config = {
        "motion_model": read_motion_model(config),
        "object_model": read_object_model(config),
        "hypothesis_model": read_hypothesis_model(config),
    }

    return TrackerConfig(**t_config)
