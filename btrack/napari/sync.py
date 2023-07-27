"""
This module contains functions for syncing widget values with TrackerConfig
values.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from btrack.config import TrackerConfig
    from btrack.napari.config import Sigmas, UnscaledTrackerConfig

from qtpy import QtWidgets

import btrack.napari.constants


def update_config_from_widgets(
    unscaled_config: UnscaledTrackerConfig,
    container: QtWidgets.QWidget,
) -> TrackerConfig:
    """Update an UnscaledTrackerConfig with the current widget values."""

    # Update MotionModel matrix scaling factors
    sigmas: Sigmas = unscaled_config.sigmas
    for matrix_name in sigmas:
        sigmas[matrix_name] = container[f"{matrix_name}_sigma"].value

    # Update TrackerConfig values
    config = unscaled_config.tracker_config
    update_method_index = container.update_method.currentIndex()

    config.update_method = update_method_index
    config.max_search_radius = container.max_search_radius.value

    # Update MotionModel values
    motion_model = config.motion_model
    motion_model.accuracy = container.accuracy.value
    motion_model.max_lost = container.max_lost.value

    # Update HypothesisModel.hypotheses values
    hypothesis_model = config.hypothesis_model
    hypothesis_model.hypotheses = [
        hypothesis
        for hypothesis in btrack.napari.constants.HYPOTHESES
        if container[hypothesis].isChecked()
    ]

    # Update HypothesisModel scaling factors
    for scaling_factor in btrack.napari.constants.HYPOTHESIS_SCALING_FACTORS:
        setattr(
            hypothesis_model, scaling_factor, container[scaling_factor].value
        )

    # Update HypothesisModel thresholds
    for threshold in btrack.napari.constants.HYPOTHESIS_THRESHOLDS:
        setattr(hypothesis_model, threshold, container[threshold].value)

    hypothesis_model.segmentation_miss_rate = (
        container.segmentation_miss_rate.value
    )

    return unscaled_config


def update_widgets_from_config(
    unscaled_config: UnscaledTrackerConfig,
    container: QtWidgets.QWidget,
) -> QtWidgets.QWidget:
    """
    Update the widgets in a container with the values in an
    UnscaledTrackerConfig.
    """

    # Update widgets from MotionModel matrix scaling factors
    sigmas: Sigmas = unscaled_config.sigmas
    for matrix_name in sigmas:
        container[f"{matrix_name}_sigma"].value = sigmas[matrix_name]

    # Update widgets from TrackerConfig values
    config = unscaled_config.tracker_config
    container.update_method.value = config.update_method.name
    container.max_search_radius.value = config.max_search_radius

    # Update widgets from MotionModel values
    motion_model = config.motion_model
    container.accuracy.value = motion_model.accuracy
    container.max_lost.value = motion_model.max_lost

    # Update widgets from HypothesisModel.hypotheses values
    hypothesis_model = config.hypothesis_model
    for hypothesis in btrack.napari.constants.HYPOTHESES:
        is_checked = hypothesis in hypothesis_model.hypotheses
        container[hypothesis].value = is_checked

    # Update widgets from HypothesisModel scaling factors
    for scaling_factor in btrack.napari.constants.HYPOTHESIS_SCALING_FACTORS:
        container[scaling_factor].value = getattr(
            hypothesis_model, scaling_factor
        )

    # Update widgets from HypothesisModel thresholds
    for threshold in btrack.napari.constants.HYPOTHESIS_THRESHOLDS:
        container[threshold].value = getattr(hypothesis_model, threshold)

    container.segmentation_miss_rate.value = (
        hypothesis_model.segmentation_miss_rate
    )

    return container
