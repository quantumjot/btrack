"""
This module contains functions for syncing widget values with TrackerConfig
values.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import btrack.napari.widgets
    from btrack.napari.config import Sigmas, UnscaledTrackerConfig

from qtpy import QtCore

import btrack.napari.constants


def update_config_from_widgets(
    unscaled_config: UnscaledTrackerConfig,
    btrack_widget: btrack.napari.widgets.BtrackWidget,
) -> UnscaledTrackerConfig:
    """Update an UnscaledTrackerConfig with the current widget values."""

    # Update MotionModel matrix scaling factors
    sigmas: Sigmas = unscaled_config.sigmas
    for matrix_name in sigmas:
        sigmas[matrix_name] = btrack_widget[f"{matrix_name}_sigma"].value()

    # Update TrackerConfig values
    config = unscaled_config.tracker_config
    update_method_index = btrack_widget.update_method.currentIndex()

    config.update_method = update_method_index
    config.max_search_radius = btrack_widget.max_search_radius.value()

    # Update MotionModel values
    motion_model = config.motion_model
    motion_model.accuracy = btrack_widget.accuracy.value()
    motion_model.max_lost = btrack_widget.max_lost.value()

    # Update HypothesisModel.hypotheses values
    hypothesis_model = config.hypothesis_model
    hypothesis_model.hypotheses = [
        hypothesis
        for i, hypothesis in enumerate(btrack.optimise.hypothesis.H_TYPES)
        if btrack_widget["hypotheses"].item(i).isSelected()
    ]

    # Update HypothesisModel scaling factors
    for scaling_factor in btrack.napari.constants.HYPOTHESIS_SCALING_FACTORS:
        setattr(
            hypothesis_model,
            scaling_factor,
            btrack_widget[scaling_factor].value(),
        )

    # Update HypothesisModel thresholds
    for threshold in btrack.napari.constants.HYPOTHESIS_THRESHOLDS:
        setattr(hypothesis_model, threshold, btrack_widget[threshold].value())

    hypothesis_model.segmentation_miss_rate = (
        btrack_widget.segmentation_miss_rate.value()
    )

    return unscaled_config


def update_widgets_from_config(
    unscaled_config: UnscaledTrackerConfig,
    btrack_widget: btrack.napari.widgets.BtrackWidget,
) -> btrack.napari.widgets.BtrackWidget:
    """
    Update the widgets in a btrack_widget with the values in an
    UnscaledTrackerConfig.
    """

    # Update widgets from MotionModel matrix scaling factors
    sigmas: Sigmas = unscaled_config.sigmas
    for matrix_name in sigmas:
        btrack_widget[f"{matrix_name}_sigma"].setValue(sigmas[matrix_name])

    # Update widgets from TrackerConfig values
    config = unscaled_config.tracker_config
    btrack_widget.update_method.setCurrentText(config.update_method.name)
    btrack_widget.max_search_radius.setValue(config.max_search_radius)

    # Update widgets from MotionModel values
    motion_model = config.motion_model
    btrack_widget.accuracy.setValue(motion_model.accuracy)
    btrack_widget.max_lost.setValue(motion_model.max_lost)

    # Update widgets from HypothesisModel.hypotheses values
    hypothesis_model = config.hypothesis_model
    for i, hypothesis in enumerate(btrack.optimise.hypothesis.H_TYPES):
        is_checked = (
            QtCore.Qt.CheckState.Checked
            if hypothesis in hypothesis_model.hypotheses
            else QtCore.Qt.CheckState.Unchecked
        )
        btrack_widget["hypotheses"].item(i).setCheckState(is_checked)

    # Update widgets from HypothesisModel scaling factors
    for scaling_factor in btrack.napari.constants.HYPOTHESIS_SCALING_FACTORS:
        new_value = getattr(hypothesis_model, scaling_factor)
        btrack_widget[scaling_factor].setValue(new_value)

    # Update widgets from HypothesisModel thresholds
    for threshold in btrack.napari.constants.HYPOTHESIS_THRESHOLDS:
        new_value = getattr(hypothesis_model, threshold)
        btrack_widget[threshold].setValue(new_value)

    btrack_widget.relax.setChecked(hypothesis_model.relax)
    btrack_widget.segmentation_miss_rate.setValue(
        hypothesis_model.segmentation_miss_rate
    )

    return btrack_widget
