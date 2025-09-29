from __future__ import annotations

import btrack.napari.constants

from qtpy import QtCore, QtWidgets


def _create_hypotheses_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
    """Create widgets for selecting which hypotheses to generate."""

    hypotheses = btrack.optimise.hypothesis.H_TYPES
    tooltips = [
        "Hypothesis that a tracklet is a false positive detection. Always required.",
        "Hypothesis that a tracklet starts at the beginning of the movie or edge of the field of view.",  # noqa: E501
        "Hypothesis that a tracklet ends at the end of the movie or edge of the field of view.",  # noqa: E501
        "Hypothesis that two tracklets should be linked together.",
        "Hypothesis that a tracklet can split into two daughter tracklets.",
        "Hypothesis that a tracklet terminates without leaving the field of view.",
        "Hypothesis that two tracklets merge into one tracklet.",
    ]

    widget = QtWidgets.QListWidget()
    widget.addItems([f"{h.replace('_', '(')})" for h in hypotheses])
    flags = QtCore.Qt.ItemFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
    for i, tooltip in enumerate(tooltips):
        widget.item(i).setFlags(flags)
        widget.item(i).setToolTip(tooltip)

    # P_FP is always required
    widget.item(hypotheses.index("P_FP")).setFlags(
        QtCore.Qt.ItemIsUserCheckable,
    )

    return {"hypotheses": ("hypotheses", widget)}


def _create_scaling_factor_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
    """Create widgets for setting the scaling factors of the HypothesisModel"""

    names = btrack.napari.constants.HYPOTHESIS_SCALING_FACTORS
    labels = [
        "λ time",
        "λ distance",
        "λ linking",
        "λ branching",
    ]
    tooltips = [
        "Scaling factor for the influence of time when determining initialization or termination hypotheses.",  # noqa: E501
        "Scaling factor for the influence of distance at the border when determining initialization or termination hypotheses.",  # noqa: E501
        "Scaling factor for the influence of track-to-track distance on linking probability.",  # noqa: E501
        "Scaling factor for the influence of cell state and position on division (mitosis/branching) probability.",  # noqa: E501
    ]

    scaling_factor_widgets = {}
    for name, label, tooltip in zip(names, labels, tooltips):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        widget.setToolTip(tooltip)
        scaling_factor_widgets[name] = (label, widget)

    return scaling_factor_widgets


def _create_threshold_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
    """Create widgets for setting thresholds for the HypothesisModel"""

    distance_threshold = QtWidgets.QDoubleSpinBox()
    distance_threshold.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    distance_threshold.setToolTip(
        "A threshold distance from the edge of the field of view to add an "
        "initialization or termination hypothesis."
    )
    widgets = {"theta_dist": ("distance threshold", distance_threshold)}

    time_threshold = QtWidgets.QDoubleSpinBox()
    time_threshold.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    time_threshold.setToolTip(
        "A threshold time from the beginning or end of movie to add "
        "an initialization or termination hypothesis."
    )
    widgets["theta_time"] = ("time threshold", time_threshold)

    apoptosis_threshold = QtWidgets.QSpinBox()
    apoptosis_threshold.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    apoptosis_threshold.setToolTip(
        "Number of apoptotic detections to be considered a genuine event.\n"
        "Detections are counted consecutively from the back of the track"
    )
    widgets["apop_thresh"] = ("apoptosis threshold", apoptosis_threshold)

    return widgets


def _create_bin_size_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
    """Create widget for setting bin sizes for the HypothesisModel"""

    distance_bin_size = QtWidgets.QDoubleSpinBox()
    distance_bin_size.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    distance_bin_size.setToolTip(
        "Isotropic spatial bin size for considering hypotheses.\n"
        "Larger bin sizes generate more hypothesese for each tracklet."
    )
    widgets = {"dist_thresh": ("distance bin size", distance_bin_size)}

    time_bin_size = QtWidgets.QDoubleSpinBox()
    time_bin_size.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    time_bin_size.setToolTip(
        "Temporal bin size for considering hypotheses.\n"
        "Larger bin sizes generate more hypothesese for each tracklet."
    )
    widgets["time_thresh"] = ("time bin size", time_bin_size)

    return widgets


def create_optimiser_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
    """Create widgets for setting parameters of the HypothesisModel"""

    widgets = {
        **_create_hypotheses_widgets(),
        **_create_scaling_factor_widgets(),
        **_create_threshold_widgets(),
        **_create_bin_size_widgets(),
    }

    segmentation_miss_rate = QtWidgets.QDoubleSpinBox()
    segmentation_miss_rate.setStepType(
        QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType
    )
    segmentation_miss_rate.setToolTip(
        "Miss rate for the segmentation.\n"
        "e.g. 1/100 segmentations incorrect gives a segmentation miss rate of 0.01."
    )
    widgets["segmentation_miss_rate"] = ("miss rate", segmentation_miss_rate)

    relax = QtWidgets.QCheckBox()
    relax.setToolTip(
        "Disable the time and distance thresholds.\n"
        "This means that tracks can initialize or terminate anywhere and"
        "at any time in the dataset."
    )
    relax.setTristate(False)
    widgets["relax"] = ("relax thresholds", relax)

    return widgets
