from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magicgui.widgets import Widget

import magicgui
import napari


def create_input_widgets() -> list[Widget]:
    """Create widgets for selecting labels layer and TrackerConfig"""

    segmentation_tooltip = (
        "Select a 'Labels' layer to use for tracking.\n"
        "To use an 'Image' layer, first convert 'Labels' by right-clicking "
        "on it in the layers list, and clicking on 'Convert to Labels'"
    )
    segmentation = magicgui.widgets.create_widget(
        annotation=napari.layers.Labels,
        name="segmentation",
        label="segmentation",
        options={"tooltip": segmentation_tooltip},
    )

    config_tooltip = (
        "Select a loaded configuration.\nNote, this will update values set below."
    )
    config = magicgui.widgets.create_widget(
        value="cell",
        name="config",
        label="config name",
        widget_type="ComboBox",
        options={
            "choices": ["cell", "particle"],
            "tooltip": config_tooltip,
        },
    )

    return [segmentation, config]


def create_update_method_widgets() -> list[Widget]:
    """Create widgets for selecting the update method"""

    update_method_tooltip = (
        "Select the update method.\n"
        "EXACT: exact calculation of Bayesian belief matrix.\n"
        "APPROXIMATE: approximate the Bayesian belief matrix. Useful for datasets with "
        "more than 1000 particles per frame."
    )
    update_method = magicgui.widgets.create_widget(
        value="EXACT",
        name="update_method",
        label="update method",
        widget_type="ComboBox",
        options={
            "choices": ["EXACT", "APPROXIMATE"],
            "tooltip": update_method_tooltip,
        },
    )

    # TODO: this widget should be hidden when the update method is set to EXACT
    max_search_radius_tooltip = (
        "The local spatial search radius (isotropic, pixels) used when the update "
        "method is 'APPROXIMATE'"
    )
    max_search_radius = magicgui.widgets.create_widget(
        value=100,
        name="max_search_radius",
        label="search radius",
        widget_type="SpinBox",
        options={"tooltip": max_search_radius_tooltip},
    )

    return [update_method, max_search_radius]


def create_control_widgets() -> list[Widget]:
    """Create widgets for running the analysis or handling I/O.

    This includes widgets for running the tracking, saving and loading
    configuration files, and resetting the widget values to those in
    the selected config."""

    names = [
        "load_config_button",
        "save_config_button",
        "reset_button",
        "call_button",
    ]
    labels = [
        "Load configuration",
        "Save configuration",
        "Reset defaults",
        "Run",
    ]
    tooltips = [
        "Load a TrackerConfig json file.",
        "Export the current configuration to a TrackerConfig json file.",
        "Reset the current configuration to the defaults stored in the corresponding json file.",  # noqa: E501
        "Run the tracking analysis with the current configuration.",
    ]

    control_buttons = []
    for name, label, tooltip in zip(names, labels, tooltips):
        widget = magicgui.widgets.create_widget(
            name=name,
            label=label,
            widget_type="PushButton",
            options={"tooltip": tooltip},
        )
        control_buttons.append(widget)

    return control_buttons
