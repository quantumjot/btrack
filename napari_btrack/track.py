from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import btrack
import napari
import numpy as np
import numpy.typing as npt
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
from pydantic import BaseModel
from PyQt5.QtWidgets import QScrollArea

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
    unscaled_matrices: Dict[str, List[float]] = field(
        default_factory=lambda: dict(
            A_cell=[
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
            A_particle=[
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
    def get_scaled_matrix(
        cls, name: str, sigma: float, cell: bool = True
    ) -> List[float]:
        if name == "A":
            if cell:
                name = "A_cell"
            else:
                name = "A_particle"
        return (np.asarray(cls().unscaled_matrices[name]) * sigma).tolist()

    @classmethod
    def get_sigma(cls, name: str, scaled_matrix: npt.NDArray[np.float64]) -> float:
        if name == "A":
            name = "A_cell"  # doesn't matter which A we use here, as [0][0] is the same
        return scaled_matrix[0][0] / cls().unscaled_matrices[name][0]


def run_tracker(
    objects: List[PyTrackObject], tracker_config: TrackerConfig
) -> Tuple[npt.NDArray, dict, dict]:
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


def html_label_widget(label: str, tag: str = "b") -> dict:
    """
    Create a HMTL label widget.
    """
    return dict(
        widget_type="Label",
        label=f"<{tag}>{label}</{tag}>",
    )


def _create_per_model_widgets(model: BaseModel) -> List[Widget]:
    """
    For a given model create a list of widgets, ignoring entries in `ignored_names`.
    The items "hypotheses" and the various matrices need customisation,
    otherwise we can use the napari default.
    """
    widgets: List[Widget] = []
    if model:
        widgets.extend([create_widget(**html_label_widget(type(model).__name__))])
        for parameter, default_value in model:
            if parameter in hidden_variable_names:
                continue
            if parameter in Matrices().names:
                # just expose the scalar sigma to user
                sigma = Matrices.get_sigma(parameter, default_value)
                widgets.extend(
                    [
                        create_widget(
                            value=sigma, name=f"{parameter}_sigma", annotation=float
                        )
                    ]
                )
            if (
                parameter == "hypotheses"
            ):  # this list should be represented as a series of checkboxes
                for choice in default_value:
                    widgets.extend(
                        [create_widget(value=True, name=choice, annotation=bool)]
                    )
            else:  # use napari default
                widgets.extend(
                    [
                        create_widget(
                            value=default_value,
                            name=parameter,
                            annotation=type(default_value),
                        )
                    ]
                )
    return widgets


def _create_napari_specific_widgets(widgets: List[Widget]) -> None:
    """
    add the widgets which interact with napari itself
    """
    widgets.append(create_widget(**html_label_widget("Segmentation")))
    widgets.append(create_widget(name="segmentation", annotation=napari.layers.Image))


def _create_pydantic_default_widgets(
    widgets: List[Widget], model_configs: List[BaseModel]
) -> None:
    """
    create the widgets which are detected automatically by napari
    """
    model_widgets = [_create_per_model_widgets(model) for model in model_configs]
    widgets.extend([item for sublist in model_widgets for item in sublist])


def _create_cell_or_particle_widget(widgets: List[Widget]) -> None:
    """Create a dropdown menu to choose between cell or particle mode."""
    widgets.extend([create_widget(**html_label_widget("Mode"))])
    widgets.extend(
        [
            create_widget(
                name="mode", value="cell", options={"choices": ["cell", "particle"]}
            )
        ]
    )


def _widgets_to_tracker_config(container: Container) -> TrackerConfig:
    motion_model_dict: Dict[str, Any] = {}
    hypothesis_model_dict = {}

    motion_model_keys = getattr(default_cell_config, "motion_model").dict().keys()
    hypothesis_model_keys = (
        getattr(default_cell_config, "hypothesis_model").dict().keys()
    )
    hypotheses = []
    for widget in container:
        # setup motion model
        if widget.name in Matrices().names:  # matrices need special treatment
            sigma = getattr(container, f"{widget.name}_sigma").value
            matrix = Matrices.get_scaled_matrix(
                widget.name, sigma, container.mode.value == "cell"
            )
            motion_model_dict[widget.name] = matrix
        else:
            if widget.name in motion_model_keys:
                motion_model_dict[widget.name] = widget.value
        # setup hypothesis model
        if widget.name in hypothesis_model_keys:
            hypothesis_model_dict[widget.name] = widget.value
        if widget.name in all_hypotheses:  # hypotheses need special treatment
            if getattr(container, widget.name).value:
                hypotheses.append(widget.name)

    # add some non-exposed default values to the motion model
    mode = getattr(container, "mode").value
    for default_name, default_value in zip(
        ["measurements", "states", "dt", "prob_not_assign", "name"],
        [3, 6, 1.0, 0.001, f"{mode}_motion"],
    ):
        motion_model_dict[default_name] = default_value

    # add some non-exposed default value to the hypothesis model
    for default_name, default_value in zip(
        ["apoptosis_rate", "eta", "name"], [0.001, 1.0e-10, f"{mode}_hypothesis"]
    ):
        hypothesis_model_dict[default_name] = default_value

    # add hypotheses to hypothesis model
    hypothesis_model_dict["hypotheses"] = hypotheses
    motion_model = MotionModel(**motion_model_dict)
    hypothesis_model = HypothesisModel(**hypothesis_model_dict)
    return TrackerConfig(motion_model=motion_model, hypothesis_model=hypothesis_model)


def _tracker_config_to_widgets(container: Container, config: TrackerConfig):
    for model in ["motion_model", "hypothesis_model", "object_model"]:
        model_config = getattr(config, model)
        if model_config:
            for parameter, value in model_config:
                if parameter in hidden_variable_names:
                    continue
                if parameter in Matrices().names:
                    sigma = Matrices.get_sigma(parameter, value)
                    getattr(container, f"{parameter}_sigma").value = sigma
                if parameter == "hypotheses":
                    for hypothesis in all_hypotheses:
                        getattr(container, hypothesis).value = False
                    for hypothesis in value:
                        getattr(container, hypothesis).value = True
                else:
                    getattr(container, parameter).value = value
    mode_is_cell = config.motion_model.A[0, 3] == 1
    print("mode is cell: ", mode_is_cell)
    container.mode.value = "cell" if mode_is_cell else "particle"


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
    _create_cell_or_particle_widget(widgets)
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
