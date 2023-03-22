from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magicgui.widgets import Container

import json
from unittest.mock import patch

import napari
import numpy as np
import numpy.typing as npt
import pytest

import btrack
import btrack.napari
import btrack.napari.main
from btrack import datasets
from btrack.datasets import cell_config, particle_config

OLD_WIDGET_LAYERS = 1
NEW_WIDGET_LAYERS = 2


def test_add_widget(make_napari_viewer):
    """Checks that the track widget can be added inside a dock widget."""

    viewer = make_napari_viewer()
    num_dw = len(list(viewer.window._dock_widgets))
    viewer.window.add_plugin_dock_widget(
        plugin_name="btrack",
        widget_name="Track",
    )

    assert len(list(viewer.window._dock_widgets)) == num_dw + 1


@pytest.fixture
def track_widget(make_napari_viewer) -> Container:
    """Provides an instance of the track widget to test"""
    make_napari_viewer()  # make sure there is a viewer available
    return btrack.napari.main.create_btrack_widget()


@pytest.mark.parametrize("config", [cell_config(), particle_config()])
def test_config_to_widgets_round_trip(track_widget, config):
    """Tests that going back and forth between
    config objects and widgets works as expected.
    """

    expected_config = btrack.config.load_config(config).json()

    unscaled_config = btrack.napari.config.UnscaledTrackerConfig(config)
    btrack.napari.sync.update_widgets_from_config(
        unscaled_config, track_widget
    )
    btrack.napari.sync.update_config_from_widgets(
        unscaled_config, track_widget
    )

    actual_config = unscaled_config.scale_config().json()

    # use json.loads to avoid failure in string comparison because e.g "100.0" != "100"
    assert json.loads(actual_config) == json.loads(expected_config)


def test_save_button(track_widget):
    """Tests that clicking the save configuration button
    triggers a call to btrack.config.save_config with expected arguments.
    """

    unscaled_config = btrack.napari.config.UnscaledTrackerConfig(cell_config())
    unscaled_config.tracker_config.name = (
        "cell"  # this is done in in the gui too
    )
    expected_config = unscaled_config.scale_config().json()

    with patch(
        "btrack.napari.widgets.save_path_dialogue_box"
    ) as save_path_dialogue_box:
        save_path_dialogue_box.return_value = "user_config.json"
        track_widget.save_config_button.clicked()

    actual_config = btrack.config.load_config("user_config.json").json()

    # use json.loads to avoid failure in string comparison because e.g "100.0" != "100"
    assert json.loads(expected_config) == json.loads(actual_config)


def test_load_config(track_widget):
    """Tests that another TrackerConfig can be loaded and made the current config."""

    # this is set to be 'cell' rather than 'Default'
    original_config_name = track_widget.config.current_choice

    with patch(
        "btrack.napari.widgets.load_path_dialogue_box"
    ) as load_path_dialogue_box:
        load_path_dialogue_box.return_value = cell_config()
        track_widget.load_config_button.clicked()

    # We didn't override the name, so it should be 'Default'
    new_config_name = track_widget.config.current_choice

    assert track_widget.config.value == "Default"
    assert new_config_name != original_config_name


def test_reset_button(track_widget):
    """Tests that clicking the reset button restores the default config values"""

    original_max_search_radius = track_widget.max_search_radius.value
    original_relax = track_widget.relax.value

    # change some widget values
    track_widget.max_search_radius.value += 10
    track_widget.relax.value = not track_widget.relax

    # click reset button - restores defaults of the currently-selected base config
    track_widget.reset_button.clicked()

    new_max_search_radius = track_widget.max_search_radius.value
    new_relax = track_widget.relax.value

    assert new_max_search_radius == original_max_search_radius
    assert new_relax == original_relax


@pytest.fixture
def simplistic_tracker_outputs() -> (
    tuple[npt.NDArray, dict[str, npt.NDArray], dict[int, list]]
):
    """Provides simplistic return values of a btrack run.

    They have the correct types and dimensions, but contain zeros.
    Useful for mocking the tracker.
    """
    n, d = 10, 3
    data = np.zeros((n, d + 1))
    properties = {"some_property": np.zeros(n)}
    graph = {0: [0]}
    return data, properties, graph


def test_run_button(track_widget, simplistic_tracker_outputs):
    """Tests that clicking the run button calls run_tracker,
    and that the napari viewer has an additional tracks layer after running.
    """
    with patch("btrack.napari.main._run_tracker") as run_tracker:
        run_tracker.return_value = simplistic_tracker_outputs
        segmentation = datasets.example_segmentation()
        track_widget.viewer.add_labels(segmentation)
        assert len(track_widget.viewer.layers) == OLD_WIDGET_LAYERS
        track_widget.call_button.clicked()
    assert run_tracker.called
    assert len(track_widget.viewer.layers) == NEW_WIDGET_LAYERS
    assert isinstance(track_widget.viewer.layers[-1], napari.layers.Tracks)
