from __future__ import annotations

from qtpy import QtWidgets


def _create_sigma_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
    """Create widgets for setting the magnitudes of the MotionModel matrices"""

    P_sigma = QtWidgets.QDoubleSpinBox()
    P_sigma.setRange(0, 500)
    P_sigma.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    P_sigma.setToolTip(
        "Magnitude of error in initial estimates.\nUsed to scale the matrix P."
    )
    P_sigma.setValue(150.0)
    widgets = {"P_sigma": ("max(<b>P</b>)", P_sigma)}

    G_sigma = QtWidgets.QDoubleSpinBox()
    G_sigma.setRange(0, 500)
    G_sigma.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    G_sigma.setToolTip("Magnitude of error in process\nUsed to scale the matrix G.")
    G_sigma.setValue(15.0)
    widgets["G_sigma"] = ("max(<b>G</b>)", G_sigma)

    R_sigma = QtWidgets.QDoubleSpinBox()
    R_sigma.setRange(0, 500)
    R_sigma.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    R_sigma.setToolTip(
        "Magnitude of error in measurements.\nUsed to scale the matrix R."
    )
    R_sigma.setValue(5.0)
    widgets["R_sigma"] = ("max(<b>R</b>)", R_sigma)

    return widgets


def create_motion_model_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
    """Create widgets for setting parameters of the MotionModel"""

    widgets = _create_sigma_widgets()

    accuracy = QtWidgets.QDoubleSpinBox()
    accuracy.setRange(0.1, 10)
    accuracy.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    accuracy.setToolTip("Integration limits for calculating probabilities")
    accuracy.setValue(7.5)
    widgets["accuracy"] = ("accuracy", accuracy)

    return widgets
