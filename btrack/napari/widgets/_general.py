from __future__ import annotations

from qtpy import QtWidgets


def create_input_widgets() -> dict[str, QtWidgets.QWidget]:
    """Create widgets for selecting labels layer and TrackerConfig"""

    # TODO: annotation=napari.layers.Labels,
    segmentation = QtWidgets.QComboBox()
    segmentation.setToolTip(
        "Select a 'Labels' layer to use for tracking.\n"
        "To use an 'Image' layer, first convert 'Labels' by right-clicking "
        "on it in the layers list, and clicking on 'Convert to Labels'"
    )
    widgets = {"segmentation": ("segmentation", segmentation)}

    config = QtWidgets.QComboBox()
    config.addItems(["cell", "particle"])
    config.setToolTip(
        "Select a loaded configuration.\n"
        "Note, this will update values set below."
    )
    widgets["config"] = ("config name", config)

    return widgets


def create_update_method_widgets() -> dict[str, QtWidgets.QWidget]:
    """Create widgets for selecting the update method"""

    update_method = QtWidgets.QComboBox()
    update_method.addItems(
        [
            "EXACT",
            "APPROXIMATE",
        ]
    )
    update_method.setToolTip(
        "Select the update method.\n"
        "EXACT: exact calculation of Bayesian belief matrix.\n"
        "APPROXIMATE: approximate the Bayesian belief matrix. Useful for datasets with "
        "more than 1000 particles per frame."
    )
    widgets = {"update_method": ("update method", update_method)}

    max_search_radius = QtWidgets.QSpinBox()
    max_search_radius.setRange(0, 1000)
    max_search_radius.setSingleStep(1)
    max_search_radius.setToolTip(
        "The local spatial search radius (isotropic, pixels) used when the update "
        "method is 'APPROXIMATE'"
    )
    max_search_radius.setWrapping(True)  # noqa: FBT003
    widgets["max_search_radius"] = ("search radius", max_search_radius)

    return widgets


def create_control_widgets() -> dict[str, QtWidgets.QWidget]:
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

    control_buttons = {}
    for name, label, tooltip in zip(names, labels, tooltips):
        widget = QtWidgets.QPushButton()
        widget.setText(label)
        widget.setToolTip(tooltip)
        control_buttons[name] = widget

    return control_buttons
