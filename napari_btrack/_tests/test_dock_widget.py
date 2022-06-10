from unittest.mock import patch

import napari
import numpy as np
import pytest
from btrack import datasets
from btrack.config import load_config
from btrack.datasets import cell_config, particle_config

from ..track import _tracker_config_to_widgets, _widgets_to_tracker_config, track


def test_add_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    num_dw = len(list(viewer.window._dock_widgets))
    viewer.window.add_function_widget(function=track)
    assert len(list(viewer.window._dock_widgets)) == num_dw + 1


@pytest.fixture
def track_widget():
    return track()


@pytest.fixture
def user_config_path():
    return "user_config.json"


@pytest.mark.parametrize("config", [cell_config(), particle_config()])
def test_config_to_widgets_round_trip(track_widget, config):
    expected_config = load_config(config)
    _tracker_config_to_widgets(track_widget, expected_config)
    actual_config = _widgets_to_tracker_config(track_widget)

    assert actual_config.json() == expected_config.json()


def test_reset_button(track_widget):
    # change config to particle
    _tracker_config_to_widgets(track_widget, load_config(particle_config()))

    # click reset button (default is cell_config)
    track_widget.reset_button.clicked()
    config_after_reset = _widgets_to_tracker_config(track_widget)

    assert config_after_reset.json() == load_config(cell_config()).json()


def test_save_button(user_config_path, track_widget):
    with patch("napari_btrack.track.save_config") as save_config:
        with patch("napari_btrack.track.get_save_path") as get_save_path:
            get_save_path.return_value = user_config_path
            track_widget.save_config_button.clicked()
    assert save_config.call_args.args[0] == user_config_path
    assert save_config.call_args.args[1].json() == load_config(cell_config()).json()


def test_load_button(user_config_path, track_widget):
    with patch("napari_btrack.track.load_config") as load_config:
        with patch("napari_btrack.track.get_load_path") as get_load_path:
            get_load_path.return_value = user_config_path
            track_widget.load_config_button.clicked()
    assert load_config.call_args.args[0] == user_config_path


@pytest.fixture
def simplistic_tracker_outputs():
    N = 10
    D = 3
    data = np.zeros((N, D + 1))
    properties = dict(some_property=np.zeros((N)))
    graph = dict()
    return data, properties, graph


def test_run_button(make_napari_viewer, track_widget, simplistic_tracker_outputs):
    with patch("napari_btrack.track.run_tracker") as run_tracker:
        run_tracker.return_value = simplistic_tracker_outputs
        viewer = make_napari_viewer()
        segmentation = datasets.example_segmentation()
        viewer.add_labels(segmentation)
        track_widget.segmentation.choices = napari.current_viewer().layers
        track_widget.segmentation.value = napari.current_viewer().layers["segmentation"]
        track_widget.call_button.clicked()
    assert run_tracker.called
    assert len(napari.current_viewer().layers) == 2
    assert type(napari.current_viewer().layers[-1]) == napari.layers.Tracks
