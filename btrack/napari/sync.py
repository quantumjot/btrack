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
    ## Retrieve model configs
    config = unscaled_config.tracker_config
    motion_model = config.motion_model
    hypothesis_model = config.hypothesis_model

    ## Update widgets from the Method tab
    config.update_method = btrack_widget.update_method.currentIndex()
    config.max_search_radius = btrack_widget.max_search_radius.value()
    motion_model.max_lost = btrack_widget.max_lost.value()
    motion_model.prob_not_assign = btrack_widget.prob_not_assign.value()
    config.enable_optimisation = (
        btrack_widget.enable_optimisation.checkState() == QtCore.Qt.CheckState.Checked
    )

    ## Update widgets from the Motion tab
    sigmas: Sigmas = unscaled_config.sigmas
    for matrix_name in sigmas:
        sigmas[matrix_name] = btrack_widget[f"{matrix_name}_sigma"].value()
    motion_model.accuracy = btrack_widget.accuracy.value()

    ## Update widgets from the Optimiser tab
    # HypothesisModel.hypotheses values
    hypothesis_model.hypotheses = [
        hypothesis
        for i, hypothesis in enumerate(btrack.optimise.hypothesis.H_TYPES)
        if btrack_widget["hypotheses"].item(i).checkState()
        == QtCore.Qt.CheckState.Checked
    ]

    # HypothesisModel scaling factors
    for scaling_factor in btrack.napari.constants.HYPOTHESIS_SCALING_FACTORS:
        setattr(
            hypothesis_model,
            scaling_factor,
            btrack_widget[scaling_factor].value(),
        )

    # HypothesisModel thresholds
    for threshold in btrack.napari.constants.HYPOTHESIS_THRESHOLDS:
        setattr(hypothesis_model, threshold, btrack_widget[threshold].value())

    # other
    hypothesis_model.segmentation_miss_rate = (
        btrack_widget.segmentation_miss_rate.value()
    )
    hypothesis_model.relax = (
        btrack_widget.relax.checkState() == QtCore.Qt.CheckState.Checked
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
    ## Retrieve model configs
    config = unscaled_config.tracker_config
    motion_model = config.motion_model
    hypothesis_model = config.hypothesis_model

    ## Update widgets from the Method tab
    btrack_widget.update_method.setCurrentText(config.update_method.name)
    btrack_widget.max_search_radius.setValue(config.max_search_radius)
    btrack_widget.max_lost.setValue(motion_model.max_lost)
    btrack_widget.prob_not_assign.setValue(motion_model.prob_not_assign)
    btrack_widget.enable_optimisation.setChecked(config.enable_optimisation)

    ## Update widgets from the Motion tab
    sigmas: Sigmas = unscaled_config.sigmas
    for matrix_name in sigmas:
        btrack_widget[f"{matrix_name}_sigma"].setValue(sigmas[matrix_name])
    btrack_widget.accuracy.setValue(motion_model.accuracy)

    ## Update widgets from the Optimiser tab
    # HypothesisModel.hypotheses values
    for i, hypothesis in enumerate(btrack.optimise.hypothesis.H_TYPES):
        is_checked = (
            QtCore.Qt.CheckState.Checked
            if hypothesis in hypothesis_model.hypotheses
            else QtCore.Qt.CheckState.Unchecked
        )
        btrack_widget["hypotheses"].item(i).setCheckState(is_checked)

    # HypothesisModel scaling factors
    for scaling_factor in btrack.napari.constants.HYPOTHESIS_SCALING_FACTORS:
        new_value = getattr(hypothesis_model, scaling_factor)
        btrack_widget[scaling_factor].setValue(new_value)

    # HypothesisModel thresholds
    for threshold in btrack.napari.constants.HYPOTHESIS_THRESHOLDS:
        new_value = getattr(hypothesis_model, threshold)
        btrack_widget[threshold].setValue(new_value)

    # other
    btrack_widget.segmentation_miss_rate.setValue(
        hypothesis_model.segmentation_miss_rate
    )
    btrack_widget.relax.setChecked(hypothesis_model.relax)

    return btrack_widget
