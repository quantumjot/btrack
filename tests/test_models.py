import json

from btrack import models, utils

CONFIG_FILE = "./models/cell_config.json"


def test_read_motion_model():
    """Test reading the motion model."""

    with open(CONFIG_FILE, "r") as config_file:
        config = json.load(config_file)["TrackerConfig"]

    model = utils.read_motion_model(config)
    assert isinstance(model, models.MotionModel)


def test_read_hypothesis_model():
    """Test reading the hypothesis model."""

    with open(CONFIG_FILE, "r") as config_file:
        config = json.load(config_file)["TrackerConfig"]

    model = utils.read_hypothesis_model(config)
    assert isinstance(model, models.HypothesisModel)
