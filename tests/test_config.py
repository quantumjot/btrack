from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pytest
from pydantic import BaseModel

import btrack

from ._utils import CONFIG_FILE


def _random_config() -> dict:
    rng = np.random.default_rng(seed=1234)
    return {
        "max_search_radius": rng.uniform(1, 100),
        "update_method": rng.choice(btrack.constants.BayesianUpdates),
        "return_kalman": bool(rng.uniform(0, 2)),
        "verbose": bool(rng.uniform(0, 2)),
        "volume": tuple([(0, rng.uniform(1, 100)) for _ in range(3)]),
    }


def _validate_config(
    cfg: Union[btrack.BayesianTracker, BaseModel], options: dict
):
    for key, value in options.items():
        cfg_value = getattr(cfg, key)
        # takes care of recursive model definintions (i.e. MotionModel inside
        # TrackerConfig).
        if isinstance(cfg_value, BaseModel):
            _validate_config(cfg_value, value)
        else:
            np.testing.assert_equal(cfg_value, value)


def test_config():
    """Test creation of a TrackerConfig object."""
    options = _random_config()
    cfg = btrack.config.TrackerConfig(**options)
    assert isinstance(cfg, btrack.config.TrackerConfig)

    for key, value in options.items():
        np.testing.assert_equal(getattr(cfg, key), value)


def test_export_config(tmp_path):
    """Test exporting the config."""
    options = _random_config()
    cfg = btrack.config.TrackerConfig(**options)

    # export it
    tmp_cfg_file = Path(tmp_path) / "config.json"
    btrack.config.save_config(tmp_cfg_file, cfg)
    assert tmp_cfg_file.exists()


def test_import_config():
    """Test loading a config from a file."""
    cfg = btrack.config.load_config(CONFIG_FILE)
    assert isinstance(cfg, btrack.config.TrackerConfig)


def test_config_tracker_setters():
    """Test configuring the tracker using setters."""
    options = _random_config()
    with btrack.BayesianTracker() as tracker:

        # use the setters to apply the comfiguration
        for key, value in options.items():
            setattr(tracker, key, value)

        # use the getters or the config
        _validate_config(tracker, options)
        _validate_config(tracker.configuration, options)


def _cfg_dict() -> Tuple[dict, dict]:
    cfg_raw = btrack.config.load_config(CONFIG_FILE)
    cfg = _random_config()
    cfg.update(cfg_raw.dict())
    assert isinstance(cfg, dict)
    return cfg, cfg


def _cfg_file() -> Tuple[Path, dict]:
    filename = CONFIG_FILE
    assert isinstance(filename, Path)
    cfg = btrack.config.load_config(filename)
    return filename, cfg.dict()


def _cfg_pydantic() -> Tuple[btrack.config.TrackerConfig, dict]:
    cfg = btrack.config.load_config(CONFIG_FILE)
    options = _random_config()
    for key, value in options.items():
        setattr(cfg, key, value)
    assert isinstance(cfg, btrack.config.TrackerConfig)
    return cfg, cfg.dict()


@pytest.mark.parametrize("get_cfg", [_cfg_file, _cfg_dict, _cfg_pydantic])
def test_config_tracker(get_cfg):
    """Test configuring the tracker from a file, dictionary or `TrackerConfig`."""
    cfg, cfg_dict = get_cfg()

    with btrack.BayesianTracker() as tracker:
        tracker.configure(cfg)
        _validate_config(tracker, cfg_dict)
