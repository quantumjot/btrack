from __future__ import annotations

from qtpy import QtWidgets


def _make_label_bold(label: str) -> str:
    """Generate html for a bold label"""

    return f"<b>{label}</b>"


def _create_sigma_widgets() -> dict[str, tuple(str, QtWidgets.QWidget)]:
    """Create widgets for setting the magnitudes of the MotionModel matrices"""

    P_sigma = QtWidgets.QDoubleSpinBox()
    P_sigma.setToolTip(
        "Magnitude of error in initial estimates.\n"
        "Used to scale the matrix P."
    )
    P_sigma.setMaximum(250)
    P_sigma.setValue(150.0)
    P_sigma.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    widgets = {"P_sigma": (f"max({_make_label_bold('P')})", P_sigma)}

    G_sigma = QtWidgets.QDoubleSpinBox()
    G_sigma.setToolTip(
        "Magnitude of error in process.\n Used to scale the matrix G."
    )
    G_sigma.setMaximum(250)
    G_sigma.setValue(15.0)
    G_sigma.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    widgets["G_sigma"] = (f"max({_make_label_bold('G')})", G_sigma)

    R_sigma = QtWidgets.QDoubleSpinBox()
    R_sigma.setToolTip(
        "Magnitude of error in measurements.\n Used to scale the matrix R."
    )
    R_sigma.setMaximum(250)
    R_sigma.setValue(5.0)
    R_sigma.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    widgets["R_sigma"] = (f"max({_make_label_bold('R')})", R_sigma)

    return widgets


def create_motion_model_widgets() -> dict[str, tuple(str, QtWidgets.QWidget)]:
    """Create widgets for setting parameters of the MotionModel"""

    widgets = _create_sigma_widgets()

    accuracy = QtWidgets.QDoubleSpinBox()
    accuracy.setToolTip("Integration limits for calculating probabilities")
    accuracy.setValue(7.5)
    accuracy.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
    widgets["accuracy"] = ("accuracy", accuracy)

    max_lost_frames = QtWidgets.QSpinBox()
    max_lost_frames.setToolTip(
        "Number of frames without observation before marking as lost"
    )
    max_lost_frames.setValue(5)
    max_lost_frames.setStepType(
        QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType
    )
    widgets["max_lost"] = ("max lost", max_lost_frames)

    return widgets
