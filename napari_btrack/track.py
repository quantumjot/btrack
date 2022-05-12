from typing import List

import btrack
from btrack import datasets
from btrack.config import load_config
from magicgui.widgets import Container, PushButton, create_widget
from pydantic import BaseModel

default_config = load_config(datasets.cell_config())


def run_tracker(objects, config_file_path):
    with btrack.BayesianTracker() as tracker:
        # configure the tracker using a config file
        tracker.configure_from_file(config_file_path)
        tracker.max_search_radius = 50

        # append the objects to be tracked
        tracker.append(objects)

        # set the volume
        tracker.volume = ((0, 1600), (0, 1200), (-1e5, 64.0))

        # track them (in interactive mode)
        tracker.track_interactive(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks in a format for napari visualization
        data, properties, graph = tracker.to_napari(ndim=2)
        return data, properties, graph


def _create_per_model_widgets(model: BaseModel) -> List[dict]:
    """
    for a given model create a list of widgets
    """
    widgets: list = []
    if model:
        for parameter, default_value in model:
            widgets.append(create_widget(value=default_value, name=parameter))
    return widgets


def track() -> Container:
    """
    Create a series of widgets programatically
    """
    default_model_configs = [
        default_config.motion_model,
        default_config.object_model,
        default_config.hypothesis_model,
    ]

    widgets: list = []
    for model in default_model_configs:
        widgets.append(_create_per_model_widgets(model))
    flattened_widgets = [item for sublist in widgets for item in sublist]

    flattened_widgets.extend(
        [
            create_widget(name="call_button", widget_type=PushButton, label="Run"),
            create_widget(name="viewer"),
            # create_widget(name="segmentation", value=None),
            create_widget(
                name="reset_button", widget_type=PushButton, label="Reset defaults"
            ),
        ]
    )

    return Container(widgets=flattened_widgets)
