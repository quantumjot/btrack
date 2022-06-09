import pytest
from btrack.config import load_config
from btrack.datasets import cell_config, particle_config

from ..track import _tracker_config_to_widgets, _widgets_to_tracker_config, track


def test_add_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    num_dw = len(list(viewer.window._dock_widgets))
    viewer.window.add_function_widget(function=track)
    assert len(list(viewer.window._dock_widgets)) == num_dw + 1


@pytest.mark.parametrize("config", [cell_config(), particle_config()])
def test_config_to_widgets_round_trip(config):
    container_widget = track()
    expected_config = load_config(config)
    _tracker_config_to_widgets(container_widget, expected_config)
    actual_config = _widgets_to_tracker_config(container_widget)

    assert actual_config.json() == expected_config.json()
