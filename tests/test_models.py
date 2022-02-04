import json

import pytest

from btrack import models, utils

CONFIG_FILE = "./models/cell_config.json"


def test_read_motion_model():
    """Test reading the motion model."""

    with open(CONFIG_FILE, "r") as config_file:
        config = json.load(config_file)["TrackerConfig"]

    model = utils.read_motion_model(config)
    assert isinstance(model, models.MotionModel)


def test_incorrect_reshape_motion_model():
    """Test that specifying the wrong matrix dimensions raise a `ValueError`."""

    with open(CONFIG_FILE, "r") as config_file:
        config = json.load(config_file)["TrackerConfig"]

    m = config["MotionModel"]["measurements"]
    s = config["MotionModel"]["states"]

    with pytest.raises(ValueError):
        config["MotionModel"]["measurements"] = m + 1  # mess up shape
        _ = utils.read_motion_model(config)

    with pytest.raises(ValueError):
        config["MotionModel"]["measurements"] = m  # mess up shape
        config["MotionModel"]["states"] = s + 1  # mess up shape
        _ = utils.read_motion_model(config)


def test_read_hypothesis_model():
    """Test reading the hypothesis model."""

    with open(CONFIG_FILE, "r") as config_file:
        config = json.load(config_file)["TrackerConfig"]

    model = utils.read_hypothesis_model(config)
    assert isinstance(model, models.HypothesisModel)
