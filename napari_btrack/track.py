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
from qtpy.QtWidgets import QScrollArea

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


@dataclass
class Matrices:
    """helper dataclass to adapt matrix representation to and from pydantic"""

    names: List[str] = field(default_factory=lambda: ["A", "H", "P", "G", "R"])
    default_sigmas: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 150.0, 15.0, 5.0]
    )
    unscaled_matrices: dict[str, List[float]] = field(
        default_factory=lambda: dict(
            A=[
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            H=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            P=[
                0.1,
                0,
                0,
                0,
                0,
                0,
                0,
                0.1,
                0,
                0,
                0,
                0,
                0,
                0,
                0.1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            G=[0.5, 0.5, 0.5, 1, 1, 1],
            R=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        )
    )

    @classmethod
    def get_scaled_matrix(cls, name: str, sigma: float) -> List[float]:
        return (asarray(cls().unscaled_matrices[name]) * sigma).tolist()

    @classmethod
    def get_sigma(cls, name: str, scaled_matrix: ndarray[float]) -> float:
        return scaled_matrix[0][0] / cls().unscaled_matrices[name][0]


def run_tracker(
    objects: List[PyTrackObject], tracker_config: TrackerConfig
) -> Tuple[ndarray, dict, dict]:
    with btrack.BayesianTracker() as tracker:
        tracker.configure(tracker_config)
        tracker.max_search_radius = 50

        # append the objects to be tracked
        tracker.append(objects)

        # set the volume
        # TODO set from objects.
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
    widgets.extend([create_widget(**html_label_widget("Control buttons"))])
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
        default_cell_config.motion_model,
        default_cell_config.object_model,
        default_cell_config.hypothesis_model,
    ]

    # create all the widgets
    _create_napari_specific_widgets(widgets)
    _create_pydantic_default_widgets(widgets, default_model_configs)
    _create_button_widgets(widgets)

    btrack_widget = Container(widgets=widgets)

    @btrack_widget.reset_button.changed.connect
    def restore_defaults() -> None:
        _tracker_config_to_widgets(btrack_widget, default_cell_config)

    @btrack_widget.call_button.changed.connect
    def run() -> None:
        config = _widgets_to_tracker_config(btrack_widget)
        segmentation = btrack_widget.segmentation.value
        segmented_objects = segmentation_to_objects(
            segmentation.data[:100, ...]
        )  # TODO
        data, properties, graph = run_tracker(segmented_objects, config)
        viewer = napari.current_viewer()
        viewer.add_tracks(
            data=data, properties=properties, graph=graph, name=f"{segmentation}_btrack"
        )

    @btrack_widget.save_config_button.changed.connect
    def save_config_to_json() -> None:
        show_file_dialog = use_app().get_obj("show_file_dialog")
        save_path = show_file_dialog(
            mode=FileDialogMode.OPTIONAL_FILE,
            caption="Specify file to save btrack configuration",
            start_path=None,
            filter="*.json",
        )
        if save_path:  # save path is None if user cancels
            save_config(save_path, _widgets_to_tracker_config(btrack_widget))

    @btrack_widget.load_config_button.changed.connect
    def load_config_from_json() -> None:
        show_file_dialog = use_app().get_obj("show_file_dialog")
        load_path = show_file_dialog(
            mode=FileDialogMode.EXISTING_FILE,
            caption="Choose JSON file containing btrack configuration",
            start_path=None,
            filter="*.json",
        )
        if load_path:  # load path is None if user cancels
            config = load_config(load_path)
            _tracker_config_to_widgets(btrack_widget, config)

    scroll = QScrollArea()
    scroll.setWidget(btrack_widget._widget._qwidget)
    btrack_widget._widget._qwidget = scroll

    return btrack_widget
