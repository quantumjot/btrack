import json
from unittest.mock import patch

import napari
import numpy as np
import numpy.typing as npt
import pytest
from magicgui.widgets import Container

from btrack import datasets
from btrack.config import load_config
from btrack.datasets import cell_config, particle_config
from napari_btrack.track import (
    _update_widgets_from_config,
    _widgets_to_tracker_config,
    track,
)


def test_add_widget(make_napari_viewer):
    """Checks that the track widget can be added inside a dock widget."""
    viewer = make_napari_viewer()
    num_dw = len(list(viewer.window._dock_widgets))
    viewer.window.add_function_widget(function=track)
    assert len(list(viewer.window._dock_widgets)) == num_dw + 1


@pytest.fixture
def track_widget(make_napari_viewer) -> Container:
    """Provides an instance of the track widget to test"""
    make_napari_viewer()  # make sure there is a viewer available
    return track()


@pytest.mark.parametrize("config", [cell_config(), particle_config()])
def test_config_to_widgets_round_trip(track_widget, config):
    """Tests that going back and forth between
    config objects and widgets works as expected.
    """
    expected_config = load_config(config)
    _update_widgets_from_config(track_widget, expected_config)
    actual_config = _widgets_to_tracker_config(track_widget)
    # use json.loads to avoid failure in string comparison because e.g "100.0" != "100"
    assert json.loads(actual_config.json()) == json.loads(expected_config.json())


@pytest.fixture
def user_config_path() -> str:
    """Provides a (dummy) string to represent a user-provided config path."""
    return "user_config.json"


def test_save_button(user_config_path, track_widget):
    """Tests that clicking the save configuration button
    triggers a call to btrack.config.save_config with expected arguments.
    """
    with patch("napari_btrack.track.save_config") as save_config, patch(
        "napari_btrack.track.get_save_path"
    ) as get_save_path:
        get_save_path.return_value = user_config_path
        track_widget.save_config_button.clicked()
    assert save_config.call_args[0][0] == user_config_path
    # use json.loads to avoid failure in string comparison because e.g "100.0" != "100"
    assert json.loads(save_config.call_args[0][1].json()) == json.loads(
        load_config(cell_config()).json()
    )


def test_load_button(user_config_path, track_widget):
    """Tests that clicking the load configuration button
    triggers a call to btrack.config.load_config with the expected argument
    """
    with patch("napari_btrack.track.load_config") as load_config, patch(
        "napari_btrack.track.get_load_path"
    ) as get_load_path:
        get_load_path.return_value = user_config_path
        track_widget.load_config_button.clicked()
    assert load_config.call_args[0][0] == user_config_path


def test_reset_button(track_widget):
    """Tests that clicking the reset button with
    particle-config-populated widgets resets to the default (i.e. cell-config)
    """
    # change config to particle
    _update_widgets_from_config(track_widget, load_config(particle_config()))

    # click reset button (default is cell_config)
    track_widget.reset_button.clicked()
    config_after_reset = _widgets_to_tracker_config(track_widget)

    # use json.loads to avoid failure in string comparison because e.g "100.0" != "100"
    assert json.loads(config_after_reset.json()) == json.loads(
        load_config(cell_config()).json()
    )


@pytest.fixture
def simplistic_tracker_outputs() -> (
    tuple[npt.NDArray, dict[str, npt.NDArray], dict[int, list]]
):
    """Provides simplistic return values of a btrack run.

    They have the correct types and dimensions, but contain zeros.
    Useful for mocking the tracker.
    """
    N, D = 10, 3
    data = np.zeros((N, D + 1))
    properties = {"some_property": np.zeros(N)}
    graph = {0: [0]}
    return data, properties, graph


def test_run_button(track_widget, simplistic_tracker_outputs):
    """Tests that clicking the run button calls run_tracker,
    and that the napari viewer has an additional tracks layer after running.
    """
    with patch("napari_btrack.track.run_tracker") as run_tracker:
        run_tracker.return_value = simplistic_tracker_outputs
        segmentation = datasets.example_segmentation()
        track_widget.viewer.add_labels(segmentation)
        assert len(track_widget.viewer.layers) == 1
        track_widget.call_button.clicked()
    assert run_tracker.called
    assert len(track_widget.viewer.layers) == 2
    assert isinstance(track_widget.viewer.layers[-1], napari.layers.Tracks)
