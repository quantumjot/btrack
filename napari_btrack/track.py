from typing import List

import btrack
import napari
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
        widgets.extend(
            [
                create_widget(value=default_value, name=parameter)
                for parameter, default_value in model
            ]
        )
    return widgets


def track() -> Container:
    """
    Create a series of widgets programatically
    """
    # the different model types
    default_model_configs = [
        default_config.motion_model,
        default_config.object_model,
        default_config.hypothesis_model,
    ]

    # initialise a list for all widgets
    widgets: list = []

    # napari-specific widgets
    widgets.append(create_widget(name="segmentation", annotation=napari.layers.Image))

    # widgets from pydantic model
    model_widgets = [
        _create_per_model_widgets(model) for model in default_model_configs
    ]
    widgets.extend([item for sublist in model_widgets for item in sublist])

    # button widgets
    widget_details = [
        ("load_config_button", "Load configuration"),
        ("save_config_button", "Save configuration"),
        ("reset_button", "Reset defaults"),
        ("call_button", "Run"),
    ]
    widgets.extend(
        [
            create_widget(name=widget_name, label=widget_label, widget_type=PushButton)
            for widget_name, widget_label in widget_details
        ]
    )

    # print(widgets.call_button)

    return Container(widgets=widgets)
