from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets


def create_logo_widgets() -> dict[str, QtWidgets.QWidget]:
    """Creates the widgets for the title, logo and documentation"""

    title = QtWidgets.QLabel("<h3>Bayesian Tracker</h3>")
    title.setAlignment(QtCore.Qt.AlignHCenter)
    widgets = {"title": title}

    logo = QtWidgets.QLabel()
    logo.setPixmap(
        QtGui.QPixmap(
            str(Path(__file__).resolve().parents[1] / "assets" / "btrack_logo.png")
        )
    )
    widgets["logo"] = logo

    docs = QtWidgets.QLabel('<a href="https://btrack.readthedocs.io">Documentation</a>')
    docs.setAlignment(QtCore.Qt.AlignHCenter)
    docs.setOpenExternalLinks(True)  # noqa: FBT003
    docs.setTextFormat(QtCore.Qt.RichText)
    docs.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
    widgets["documentation"] = docs

    return widgets


def create_input_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
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
        "Select a loaded configuration.\nNote, this will update values set below."
    )
    widgets["config"] = ("config name", config)

    return widgets


def create_basic_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
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

    max_search_radius = QtWidgets.QDoubleSpinBox()
    max_search_radius.setRange(0, 1000)
    max_search_radius.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    max_search_radius.setToolTip(
        "The local spatial search radius (isotropic, pixels) used when the update "
        "method is 'APPROXIMATE'"
    )
    max_search_radius.setWrapping(True)  # noqa: FBT003
    widgets["max_search_radius"] = ("search radius", max_search_radius)

    max_lost_frames = QtWidgets.QSpinBox()
    max_lost_frames.setRange(0, 10)
    max_lost_frames.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    max_lost_frames.setToolTip(
        "Number of frames without observation before marking as lost"
    )
    widgets["max_lost"] = ("max lost", max_lost_frames)

    not_assign = QtWidgets.QDoubleSpinBox()
    not_assign.setDecimals(3)
    not_assign.setRange(0, 1)
    not_assign.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    not_assign.setToolTip("Default probability to not assign a track")
    widgets["prob_not_assign"] = (
        "<b>P</b>(not track)",
        not_assign,
    )

    optimise = QtWidgets.QCheckBox()
    optimise.setChecked(True)  # noqa: FBT003
    optimise.setToolTip(
        "Enable the track optimisation.\n"
        "This means that tracks will be optimised using the hypotheses"
        "specified in the optimiser tab."
    )
    optimise.setTristate(False)  # noqa: FBT003
    widgets["enable_optimisation"] = ("enable optimisation", optimise)

    return widgets


def create_config_widgets() -> dict[str, QtWidgets.QWidget]:
    """Create widgets for running the analysis or handling I/O.

    This includes widgets for running the tracking, saving and loading
    configuration files, and resetting the widget values to those in
    the selected config."""

    names = [
        "load_config_button",
        "save_config_button",
        "reset_button",
    ]
    labels = [
        "Load configuration",
        "Save configuration",
        "Reset defaults",
    ]
    tooltips = [
        "Load a TrackerConfig json file.",
        "Export the current configuration to a TrackerConfig json file.",
        "Reset the current configuration to the defaults stored in the corresponding json file.",  # noqa: E501
    ]

    widgets = {}
    for name, label, tooltip in zip(names, labels, tooltips):
        widget = QtWidgets.QPushButton()
        widget.setText(label)
        widget.setToolTip(tooltip)
        widgets[name] = widget

    return widgets


def create_track_widgets() -> dict[str, QtWidgets.QWidget]:
    call_button = QtWidgets.QPushButton("Track")
    call_button.setToolTip(
        "Run the tracking analysis with the current configuration.",
    )

    return {"call_button": call_button}
