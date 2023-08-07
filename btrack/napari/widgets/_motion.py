from __future__ import annotations

from qtpy import QtWidgets


def _create_sigma_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
    """Create widgets for setting the magnitudes of the MotionModel matrices"""

    P_sigma = QtWidgets.QDoubleSpinBox()
    P_sigma.setToolTip(
        "Magnitude of error in initial estimates.\n"
        "Used to scale the matrix P."
    )
    P_sigma.setMaximum(250)
    P_sigma.setValue(150.0)
    P_sigma.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    widgets = {"P_sigma": ("max(<b>P</b>)", P_sigma)}

    G_sigma = QtWidgets.QDoubleSpinBox()
    G_sigma.setToolTip(
        "Magnitude of error in process.\n Used to scale the matrix G."
    )
    G_sigma.setMaximum(250)
    G_sigma.setValue(15.0)
    G_sigma.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    widgets["G_sigma"] = ("max(<b>G</b>)", G_sigma)

    R_sigma = QtWidgets.QDoubleSpinBox()
    R_sigma.setToolTip(
        "Magnitude of error in measurements.\n Used to scale the matrix R."
    )
    R_sigma.setMaximum(250)
    R_sigma.setValue(5.0)
    R_sigma.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    widgets["R_sigma"] = ("max(<b>R</b>)", R_sigma)

    return widgets


def create_motion_model_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
    """Create widgets for setting parameters of the MotionModel"""

    widgets = _create_sigma_widgets()

    accuracy = QtWidgets.QDoubleSpinBox()
    accuracy.setToolTip("Integration limits for calculating probabilities")
    accuracy.setValue(7.5)
    accuracy.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    widgets["accuracy"] = ("accuracy", accuracy)

    return widgets
