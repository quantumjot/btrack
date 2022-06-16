from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import btrack
import napari
import numpy as np
import numpy.typing as npt
from btrack import datasets
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
HIDDEN_VARIABLE_NAMES = [
    "name",
    "measurements",
    "states",
    "dt",
    "apoptosis_rate",
    "prob_not_assign",
    "eta",
]
ALL_HYPOTHESES = ["P_FP", "P_init", "P_term", "P_link", "P_branch", "P_dead"]


@dataclass
class Matrices:
    """A helper dataclass to adapt matrix representation to and from pydantic.
    This is needed because TrackerConfig stores "scaled" matrices, i.e.
    doesn't store sigma and the "unscaled" matrix separately.
    """

    names: List[str] = field(default_factory=lambda: ["A", "H", "P", "G", "R"])
    default_sigmas: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 150.0, 15.0, 5.0]
    )
    unscaled_matrices: Dict[str, npt.NDArray[np.float64]] = field(
        default_factory=lambda: dict(
            A_cell=np.array(
                [
                    [1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            ),
            A_particle=np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            ),
            H=np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]),
            P=np.array(
                [
                    [0.1, 0, 0, 0, 0, 0],
                    [0, 0.1, 0, 0, 0, 0],
                    [0, 0, 0.1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            ),
            G=np.array([[0.5, 0.5, 0.5, 1, 1, 1]]),
            R=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        )
    )

    @classmethod
    def get_scaled_matrix(
        cls, name: str, *, sigma: float, use_cell_config: bool = True
    ) -> List[float]:
        """Returns the scaled version (i.e. the unscaled matrix multiplied by sigma)
        of the matrix.

        Keyword arguments:
        name -- the matrix name (can be one of A, H, P, G, R)
        sigma -- the factor to scale the matrix entries with
        cell -- whether to use cell config matrices or not (default true)
        """
        if name == "A":
            if use_cell_config:
                name = "A_cell"
            else:
                name = "A_particle"
        return (np.asarray(cls().unscaled_matrices[name]) * sigma).tolist()

    @classmethod
    def get_sigma(cls, name: str, scaled_matrix: npt.NDArray[np.float64]) -> float:
        """Returns the factor sigma which is the multiplier between the given scaled
        matrix and the unscaled matrix of the given name.

        Note: The calculation is done with the top-left entry of the matrix,
        and all other entries are ignored.

        Keyword arguments:
        name -- the matrix name (can be one of A, H, P, G, R)
        scaled_matrix -- the scaled matrix to find sigma from.
        """
        if name == "A":
            name = "A_cell"  # doesn't matter which A we use here, as [0][0] is the same
        return scaled_matrix[0][0] / cls().unscaled_matrices[name][0][0]


def run_tracker(
    segmentation: Union[napari.layers.Image, napari.layers.Labels],
    tracker_config: TrackerConfig,
) -> Tuple[npt.NDArray, dict, dict]:
    """
    Runs BayesianTracker with given segmentation and configuration.
    """
    with btrack.BayesianTracker() as tracker:
        tracker.configure(tracker_config)

        # append the objects to be tracked
        segmented_objects = segmentation_to_objects(segmentation.data)
        tracker.append(segmented_objects)

        # set the volume
        segmentation_size = segmentation.level_shapes[0]
        # btrack order of dimensions is XY(Z)
        # napari order of dimensions is T(Z)XY
        # so we ignore the first entry and then iterate backwards
        tracker.volume = tuple([(0, s) for s in segmentation_size[1:][::-1]])

        # track them (in interactive mode)
        tracker.track_interactive(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks in a format for napari visualization
        data, properties, graph = tracker.to_napari(ndim=2)
        return data, properties, graph


def get_save_path():
    """Helper function to open a save configuration file dialog."""
    show_file_dialog = use_app().get_obj("show_file_dialog")
    save_path = show_file_dialog(
        mode=FileDialogMode.OPTIONAL_FILE,
        caption="Specify file to save btrack configuration",
        start_path=None,
        filter="*.json",
    )
    return save_path


def get_load_path():
    """Helper function to open a load configuration file dialog."""
    show_file_dialog = use_app().get_obj("show_file_dialog")
    load_path = show_file_dialog(
        mode=FileDialogMode.EXISTING_FILE,
        caption="Choose JSON file containing btrack configuration",
        start_path=None,
        filter="*.json",
    )
    return load_path


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
    For a given model create the required list of widgets.
    The items "hypotheses" and the various matrices need customisation,
    otherwise we can use the napari default.
    """
    widgets: List[Widget] = []
    widgets.append(create_widget(**html_label_widget(type(model).__name__)))
    for parameter, default_value in model:
        if parameter in HIDDEN_VARIABLE_NAMES:
            continue
        if parameter in Matrices().names:
            # only expose the scalar sigma to user
            sigma = Matrices.get_sigma(parameter, default_value)
            widgets.append(
                create_widget(value=sigma, name=f"{parameter}_sigma", annotation=float)
            )
        if parameter == "hypotheses":
            # the hypothesis list should be represented as a series of checkboxes
            widgets.extend(
                [
                    create_widget(
                        value=(choice in default_value), name=choice, annotation=bool
                    )
                    for choice in ALL_HYPOTHESES
                ]
            )
        else:  # use napari default
            widgets.append(
                create_widget(
                    value=default_value, name=parameter, annotation=type(default_value)
                )
            )
    return widgets


def _create_napari_specific_widgets(widgets: List[Widget]) -> None:
    """
    Add the widgets which interact with napari itself
    """
    widgets.append(create_widget(**html_label_widget("Segmentation")))
    segmentation_widget = create_widget(
        name="segmentation",
        annotation=napari.layers.Labels,
        options=dict(
            tooltip=(
                "Should be a Labels layer. Convert an Image to Labels by right-clicking"
                "on it in the layers list, and clicking on 'Convert to Labels'"
            ),
        ),
    )
    widgets.append(segmentation_widget)


def _create_pydantic_default_widgets(
    widgets: List[Widget], config: TrackerConfig
) -> None:
    """
    Create the widgets which have a tracker config equivalent.
    """
    widgets.append(
        create_widget(name="max_search_radius", value=config.max_search_radius)
    )
    model_configs = [config.motion_model, config.hypothesis_model]
    model_widgets = [_create_per_model_widgets(model) for model in model_configs]
    widgets.extend([item for sublist in model_widgets for item in sublist])


def _create_cell_or_particle_widget(widgets: List[Widget]) -> None:
    """Create a dropdown menu to choose between cell or particle mode."""
    widgets.append(create_widget(**html_label_widget("Mode")))
    widgets.append(
        create_widget(
            name="mode", value="cell", options={"choices": ["cell", "particle"]}
        )
    )


def _widgets_to_tracker_config(container: Container) -> TrackerConfig:
    """Helper function to convert from the widgets to a tracker configuration."""
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
                widget.name,
                sigma=sigma,
                use_cell_config=(container.mode.value == "cell"),
            )
            motion_model_dict[widget.name] = matrix
        else:
            if widget.name in motion_model_keys:
                motion_model_dict[widget.name] = widget.value
        # setup hypothesis model
        if widget.name in hypothesis_model_keys:
            hypothesis_model_dict[widget.name] = widget.value
        if widget.name in ALL_HYPOTHESES:  # hypotheses need special treatment
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

    # add parameters outside the internal models
    max_search_radius = getattr(container, "max_search_radius").value
    return TrackerConfig(
        max_search_radius=max_search_radius,
        motion_model=motion_model,
        hypothesis_model=hypothesis_model,
    )


def _update_widgets_from_config(container: Container, config: TrackerConfig) -> None:
    """Helper function to update a container's widgets
    with the values in a given tracker config.
    """
    getattr(container, "max_search_radius").value = config.max_search_radius
    for model in ["motion_model", "hypothesis_model", "object_model"]:
        model_config = getattr(config, model)
        if model_config:
            for parameter, value in model_config:
                if parameter in HIDDEN_VARIABLE_NAMES:
                    continue
                if parameter in Matrices().names:
                    sigma = Matrices.get_sigma(parameter, value)
                    getattr(container, f"{parameter}_sigma").value = sigma
                if parameter == "hypotheses":
                    for hypothesis in ALL_HYPOTHESES:
                        getattr(container, hypothesis).value = hypothesis in value
                else:
                    getattr(container, parameter).value = value
    # we can determine whether we are in particle or cell mode
    # by checking whether the 4th entry of the first row of the
    # A matrix is 1 or 0 (1 for cell mode)
    mode_is_cell = config.motion_model.A[0, 3] == 1
    print("mode is cell: ", mode_is_cell)
    container.mode.value = "cell" if mode_is_cell else "particle"


def _create_button_widgets(widgets: List[Widget]) -> None:
    """Create the set of button widgets needed:
    run, save/load configuration and reset."""
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
    widgets.append(create_widget(**html_label_widget("Control buttons")))
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

    # create all the widgets
    _create_napari_specific_widgets(widgets)
    _create_cell_or_particle_widget(widgets)
    _create_pydantic_default_widgets(widgets, default_cell_config)
    _create_button_widgets(widgets)

    btrack_widget = Container(widgets=widgets)
    btrack_widget.viewer = napari.current_viewer()

    @btrack_widget.reset_button.changed.connect
    def restore_defaults() -> None:
        _update_widgets_from_config(btrack_widget, default_cell_config)

    @btrack_widget.call_button.changed.connect
    def run() -> None:
        config = _widgets_to_tracker_config(btrack_widget)
        segmentation = btrack_widget.segmentation.value
        data, properties, graph = run_tracker(segmentation, config)
        btrack_widget.viewer.add_tracks(
            data=data, properties=properties, graph=graph, name=f"{segmentation}_btrack"
        )

    @btrack_widget.save_config_button.changed.connect
    def save_config_to_json() -> None:
        save_path = get_save_path()
        if save_path:  # save path is None if user cancels
            save_config(save_path, _widgets_to_tracker_config(btrack_widget))

    @btrack_widget.load_config_button.changed.connect
    def load_config_from_json() -> None:
        load_path = get_load_path()
        if load_path:  # load path is None if user cancels
            config = load_config(load_path)
            _update_widgets_from_config(btrack_widget, config)

    scroll = QScrollArea()
    scroll.setWidget(btrack_widget._widget._qwidget)
    btrack_widget._widget._qwidget = scroll

    return btrack_widget
