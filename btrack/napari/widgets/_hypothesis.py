from __future__ import annotations

from qtpy import QtWidgets

import btrack.napari.constants


def _create_hypotheses_widgets() -> dict[str, QtWidgets.QWidget]:
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

    hypotheses_widgets = {}
    for hypothesis, tooltip in zip(hypotheses, tooltips):
        widget = QtWidgets.QCheckBox()
        widget.setCheckState(True)  # noqa: FBT003
        widget.setToolTip(tooltip)
        widget.setTristate(False)  # noqa: FBT003
        hypotheses_widgets[hypothesis] = (hypothesis, widget)

    # P_FP is always required
    P_FP_hypothesis = hypotheses_widgets["P_FP"][1]
    P_FP_hypothesis.enabled = False

    # P_merge should be disabled by default
    P_merge_hypothesis = hypotheses_widgets["P_merge"][1]
    P_merge_hypothesis.value = False

    return hypotheses_widgets


def _create_scaling_factor_widgets() -> dict[str, QtWidgets.QWidget]:
    """Create widgets for setting the scaling factors of the HypothesisModel"""

    widget_values = [5.0, 3.0, 10.0, 50.0]
    names = [
        "lambda_time",
        "lambda_dist",
        "lambda_link",
        "lambda_branch",
    ]
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
    for value, name, label, tooltip in zip(
        widget_values, names, labels, tooltips
    ):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setToolTip(tooltip)
        widget.setValue(value)
        scaling_factor_widgets[name] = (label, widget)

    return scaling_factor_widgets


def _create_threshold_widgets() -> dict[str, QtWidgets.QWidget]:
    """Create widgets for setting thresholds for the HypothesisModel"""

    distance_threshold = QtWidgets.QDoubleSpinBox()
    distance_threshold.setToolTip(
        "A threshold distance from the edge of the field of view to add an "
        "initialization or termination hypothesis."
    )
    distance_threshold.setValue(20.0)
    widgets = {"theta_dist": ("distance threshold", distance_threshold)}

    time_threshold = QtWidgets.QDoubleSpinBox()
    time_threshold.setToolTip(
        "A threshold time from the beginning or end of movie to add "
        "an initialization or termination hypothesis."
    )
    time_threshold.setValue(5.0)
    widgets["theta_time"] = ("time threshold", time_threshold)

    apoptosis_threshold = QtWidgets.QSpinBox()
    apoptosis_threshold.setToolTip(
        "Number of apoptotic detections to be considered a genuine event.\n"
        "Detections are counted consecutively from the back of the track"
    )
    apoptosis_threshold.setValue(5)
    widgets["apop_thresh"] = ("apoptosis threshold", apoptosis_threshold)

    return widgets


def _create_bin_size_widgets() -> dict[str, QtWidgets.QWidget]:
    """Create widget for setting bin sizes for the HypothesisModel"""

    distance_bin_size = QtWidgets.QDoubleSpinBox()
    distance_bin_size.setToolTip(
        "Isotropic spatial bin size for considering hypotheses.\n"
        "Larger bin sizes generate more hypothesese for each tracklet."
    )
    distance_bin_size.setValue(40.0)
    widgets = {"dist_thresh": ("distance bin size", distance_bin_size)}

    time_bin_size = QtWidgets.QDoubleSpinBox()
    time_bin_size.setToolTip(
        "Temporal bin size for considering hypotheses.\n"
        "Larger bin sizes generate more hypothesese for each tracklet."
    )
    time_bin_size.setValue(2.0)
    widgets["time_thresh"] = ("time bin size", time_bin_size)

    return widgets


def create_hypothesis_model_widgets() -> dict[str, QtWidgets.QWidget]:
    """Create widgets for setting parameters of the MotionModel"""

    hypothesis_model_label = QtWidgets.QLabel()
    widgets = {
        "hypothesis": ("<b>Hypothesis model</b>", hypothesis_model_label)
    }

    widgets |= (
        _create_hypotheses_widgets()
        | _create_scaling_factor_widgets()
        | _create_threshold_widgets()
        | _create_bin_size_widgets()
    )

    segmentation_miss_rate = QtWidgets.QDoubleSpinBox()
    segmentation_miss_rate.setToolTip(
        "Miss rate for the segmentation.\n"
        "e.g. 1/100 segmentations incorrect gives a segmentation miss rate of 0.01."
    )
    segmentation_miss_rate.setValue(0.1)
    widgets["segmentation_miss_rate"] = ("miss rate", segmentation_miss_rate)

    relax = QtWidgets.QCheckBox()
    relax.setCheckState(True)  # noqa: FBT003
    relax.setToolTip(
        "Disable the time and distance thresholds.\n"
        "This means that tracks can initialize or terminate anywhere and"
        "at any time in the dataset."
    )
    relax.setTristate(False)  # noqa: FBT003
    widgets["relax"] = ("relax thresholds", relax)

    return widgets
