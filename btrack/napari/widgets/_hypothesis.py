from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magicgui.widgets import Widget

import magicgui

import napari_btrack.constants


def _create_hypotheses_widgets() -> list[Widget]:
    """Create widgets for selecting which hypotheses to generate."""

    hypotheses = napari_btrack.constants.HYPOTHESES
    tooltips = [
        "Hypothesis that a tracklet is a false positive detection. Always required.",
        "Hypothesis that a tracklet starts at the beginning of the movie or edge of the field of view.",  # noqa: E501
        "Hypothesis that a tracklet ends at the end of the movie or edge of the field of view.",  # noqa: E501
        "Hypothesis that two tracklets should be linked together.",
        "Hypothesis that a tracklet can split into two daughter tracklets.",
        "Hypothesis that a tracklet terminates without leaving the field of view.",
        "Hypothesis that two tracklets merge into one tracklet.",
    ]

    hypotheses_widgets = []
    for hypothesis, tooltip in zip(hypotheses, tooltips):
        widget = magicgui.widgets.create_widget(
            value=True,
            name=hypothesis,
            label=hypothesis,
            widget_type="CheckBox",
            options={"tooltip": tooltip},
        )
        hypotheses_widgets.append(widget)

    # P_FP is always required
    P_FP_hypothesis = hypotheses_widgets[0]
    P_FP_hypothesis.enabled = False

    # P_merge should be disabled by default
    P_merge_hypothesis = hypotheses_widgets[-1]
    P_merge_hypothesis.value = False

    return hypotheses_widgets


def _create_scaling_factor_widgets() -> list[Widget]:
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

    scaling_factor_widgets = []
    for value, name, label, tooltip in zip(widget_values, names, labels, tooltips):
        widget = magicgui.widgets.create_widget(
            value=value,
            name=name,
            label=label,
            widget_type="FloatSpinBox",
            options={"tooltip": tooltip},
        )
        scaling_factor_widgets.append(widget)

    return scaling_factor_widgets


def _create_threshold_widgets() -> list[Widget]:
    """Create widgets for setting thresholds for the HypothesisModel"""

    distance_threshold_tooltip = (
        "A threshold distance from the edge of the field of view to add an "
        "initialization or termination hypothesis."
    )
    distance_threshold = magicgui.widgets.create_widget(
        value=20.0,
        name="theta_dist",
        label="distance threshold",
        widget_type="FloatSpinBox",
        options={"tooltip": distance_threshold_tooltip},
    )

    time_threshold_tooltip = (
        "A threshold time from the beginning or end of movie to add "
        "an initialization or termination hypothesis."
    )
    time_threshold = magicgui.widgets.create_widget(
        value=5.0,
        name="theta_time",
        label="time threshold",
        widget_type="FloatSpinBox",
        options={"tooltip": time_threshold_tooltip},
    )

    apoptosis_threshold_tooltip = (
        "Number of apoptotic detections to be considered a genuine event.\n"
        "Detections are counted consecutively from the back of the track"
    )
    apoptosis_threshold = magicgui.widgets.create_widget(
        value=5,
        name="apop_thresh",
        label="apoptosis threshold",
        widget_type="SpinBox",
        options={"tooltip": apoptosis_threshold_tooltip},
    )

    return [
        distance_threshold,
        time_threshold,
        apoptosis_threshold,
    ]


def _create_bin_size_widgets() -> list[Widget]:
    """Create widget for setting bin sizes for the HypothesisModel"""

    distance_bin_size_tooltip = (
        "Isotropic spatial bin size for considering hypotheses.\n"
        "Larger bin sizes generate more hypothesese for each tracklet."
    )
    distance_bin_size = magicgui.widgets.create_widget(
        value=40.0,
        name="dist_thresh",
        label="distance bin size",
        widget_type="FloatSpinBox",
        options={"tooltip": distance_bin_size_tooltip},
    )

    time_bin_size_tooltip = (
        "Temporal bin size for considering hypotheses.\n"
        "Larger bin sizes generate more hypothesese for each tracklet."
    )
    time_bin_size = magicgui.widgets.create_widget(
        value=2.0,
        name="time_thresh",
        label="time bin size",
        widget_type="FloatSpinBox",
        options={"tooltip": time_bin_size_tooltip},
    )

    return [
        distance_bin_size,
        time_bin_size,
    ]


def create_hypothesis_model_widgets() -> list[Widget]:
    """Create widgets for setting parameters of the MotionModel"""

    hypothesis_model_label = magicgui.widgets.create_widget(
        label="<b>Hypothesis model</b>",  # bold label
        widget_type="Label",
        gui_only=True,
    )

    hypotheses_widgets = _create_hypotheses_widgets()
    scaling_factor_widgets = _create_scaling_factor_widgets()
    threshold_widgets = _create_threshold_widgets()
    bin_size_widgets = _create_bin_size_widgets()

    segmentation_miss_rate_tooltip = (
        "Miss rate for the segmentation.\n"
        "e.g. 1/100 segmentations incorrect gives a segmentation miss rate of 0.01."
    )
    segmentation_miss_rate = magicgui.widgets.create_widget(
        value=0.1,
        name="segmentation_miss_rate",
        label="miss rate",
        widget_type="FloatSpinBox",
        options={"tooltip": segmentation_miss_rate_tooltip},
    )

    relax_tooltip = (
        "Disable the time and distance thresholds.\n"
        "This means that tracks can initialize or terminate anywhere and"
        "at any time in the dataset."
    )
    relax = magicgui.widgets.create_widget(
        value=True,
        name="relax",
        label="relax thresholds",
        widget_type="CheckBox",
        options={"tooltip": relax_tooltip},
    )

    return [
        hypothesis_model_label,
        *hypotheses_widgets,
        *scaling_factor_widgets,
        *threshold_widgets,
        *bin_size_widgets,
        segmentation_miss_rate,
        relax,
    ]
