from dataclasses import dataclass, field
from typing import List, Tuple

import btrack
import napari
from btrack import datasets
from btrack.btypes import PyTrackObject
from btrack.config import (
    HypothesisModel,
    MotionModel,
    TrackerConfig,
    load_config,
    save_config,
)
from btrack.utils import segmentation_to_objects
from magicgui.application import use_app
from magicgui.types import FileDialogMode
from magicgui.widgets import Container, PushButton, Widget, create_widget
from numpy import asarray, ndarray
from pydantic import BaseModel

default_cell_config = load_config(datasets.cell_config())

# widgets for which the default widget type is incorrect
hidden_variable_names = [
    "name",
    "measurements",
    "states",
    "dt",
    "apoptosis_rate",
    "prob_not_assign",
    "eta",
]
all_hypotheses = ["P_FP", "P_init", "P_term", "P_link", "P_branch", "P_dead"]

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


def _create_per_model_widgets(
    model: BaseModel, non_standard_widget_names: List[str]
) -> List[Widget]:
    """
    for a given model create a list of widgets but skip the
    ones in which napari guesses the incorrect type
    """
    widgets: List[Widget] = []
    if model:
        widgets.extend(
            [
                create_widget(value=default_value, name=parameter)
                for parameter, default_value in model
                if parameter not in non_standard_widget_names
            ]
        )
    return widgets


def _create_napari_specific_widgets(widgets: List[Widget]) -> None:
    """
    add the widgets which interact with napari itself
    """
    widgets.append(create_widget(name="segmentation", annotation=napari.layers.Image))


def _create_pydantic_default_widgets(
    widgets: List[Widget],
    model_configs: List[BaseModel],
    *,
    non_standard_widget_names: List[str] = [],
) -> None:
    """
    create the widgets which are detected automatically by napari
    """
    model_widgets = [
        _create_per_model_widgets(model, non_standard_widget_names)
        for model in model_configs
    ]
    widgets.extend([item for sublist in model_widgets for item in sublist])


def _create_button_widgets(widgets: List[Widget]) -> None:
    """
    create the set of button widgets at the bottom of the widget with
    appropriate callbacks to enable functionality
    """
    widget_names = [
        "load_config_button",
        "save_config_button",
        "reset_button",
        "call_button",
    ]
    widget_labels = [
        "Load configuration",
        "Save configuration",
        "Reset defaults",
        "Run",
    ]
    widgets.extend(
        [
            create_widget(name=widget_name, label=widget_label, widget_type=PushButton)
            for widget_name, widget_label in zip(widget_names, widget_labels)
        ]
    )


def track() -> Container:
    """
    Create a series of widgets programatically
    """
    # initialise a list for all widgets
    widgets: list = []

    # the different model types
    default_model_configs = [
        default_config.motion_model,
        default_config.object_model,
        default_config.hypothesis_model,
    ]

    # widgets for which the default widget type is incorrect
    non_standard_widgets = [
        create_widget(
            name="hypotheses",
            value=getattr(default_config.hypothesis_model, "hypotheses")[0],
            options=dict(
                choices=getattr(default_config.hypothesis_model, "hypotheses")
            ),
        )
    ]

    # create all the widgets
    _create_napari_specific_widgets(widgets)
    _create_pydantic_default_widgets(
        widgets,
        default_model_configs,
        non_standard_widget_names=[w.name for w in non_standard_widgets],
    )
    widgets.extend(non_standard_widgets)
    _create_button_widgets(widgets)

    btrack_widget = Container(widgets=widgets)

    @btrack_widget.reset_button.changed.connect
    def restore_defaults():
        # treat hypotheses different for now
        btrack_widget.hypotheses.value = getattr(
            default_config.hypothesis_model, "hypotheses"
        )[0]

        for model in default_model_configs:
            if model:
                for parameter, default_value in model:
                    if parameter == "hypotheses":
                        btrack_widget[parameter].value = default_value

    @btrack_widget.call_button.changed.connect
    def run():
        segmentation = btrack_widget.segmentation.value
        segmented_objects = segmentation_to_objects(segmentation.data[:100, ...])
        data, properties, graph = run_tracker(segmented_objects, datasets.cell_config())
        viewer = napari.current_viewer()
        viewer.add_tracks(
            data=data, properties=properties, graph=graph, name=f"{segmentation}_btrack"
        )

    @btrack_widget.save_config_button.changed.connect
    def save_config_to_json():
        print("save config")

    @btrack_widget.load_config_button.changed.connect
    def load_config_from_json():
        print("load config")

    return btrack_widget
