import json

import numpy as np
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


def test_missing_process_covariance_motion_model():
    """Test missing process covariance in motion model."""
    with open(CONFIG_FILE, "r") as config_file:
        config = json.load(config_file)["TrackerConfig"]

    # delete the G matrix
    del config["MotionModel"]["G"]

    with pytest.raises(ValueError):
        _ = utils.read_motion_model(config)


def test_specifying_process_covariance_Q_motion_model():
    """Test specifying process covariance in motion model using Q."""
    with open(CONFIG_FILE, "r") as config_file:
        config = json.load(config_file)["TrackerConfig"]

    assert "G" in config["MotionModel"]
    assert "Q" not in config["MotionModel"]

    model = utils.read_motion_model(config)

    # delete the G matrix
    test_config = dict(config)
    sigma = test_config["MotionModel"]["G"]["sigma"]
    del test_config["MotionModel"]["G"]

    # create a test configuration specifying a Q matrix instead of a G matrix
    test_config["MotionModel"]["Q"] = {
        "sigma": sigma,
        "matrix": (model.Q.ravel() / sigma).tolist(),
    }

    assert "G" not in test_config["MotionModel"]
    assert "Q" in test_config["MotionModel"]

    test_model = utils.read_motion_model(test_config)

    np.testing.assert_equal(model.Q, test_model.Q)
