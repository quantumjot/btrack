from __future__ import annotations

from qtpy import QtWidgets


def _make_label_bold(label: str) -> str:
    """Generate html for a bold label"""

    return f"<b>{label}</b>"


def _create_sigma_widgets() -> dict[str, QtWidgets.QWidget]:
    """Create widgets for setting the magnitudes of the MotionModel matrices"""

    P_sigma = QtWidgets.QDoubleSpinBox()
    P_sigma.setName(f"max({_make_label_bold('P')})")
    P_sigma.setToolTip(
        "Magnitude of error in initial estimates.\n"
        "Used to scale the matrix P."
    )
    P_sigma.setValue(150.0)
    widgets = {"P_sigma": P_sigma}

    G_sigma = QtWidgets.QDoubleSpinBox()
    G_sigma.setName(f"max({_make_label_bold('G')})")
    G_sigma.setToolTip(
        "Magnitude of error in process.\n Used to scale the matrix G."
    )
    G_sigma.setValue(15.0)
    widgets["G_sigma"] = G_sigma

    R_sigma = QtWidgets.QDoubleSpinBox()
    R_sigma.setName(f"max({_make_label_bold('R')})")
    R_sigma.setToolTip(
        "Magnitude of error in measurements.\n Used to scale the matrix R."
    )
    R_sigma.setValue(5.0)
    widgets["R_sigma"] = R_sigma

    return widgets


def create_motion_model_widgets() -> dict[str, QtWidgets.QWidget]:
    """Create widgets for setting parameters of the MotionModel"""

    motion_model_label = QtWidgets.QLabel()
    motion_model_label.setName(_make_label_bold("Motion model"))
    widgets = {"motion_model": motion_model_label}

    widgets |= _create_sigma_widgets()

    accuracy = QtWidgets.QDoubleSpinBox()
    accuracy.setName("accuracy")
    accuracy.setToolTip("Integration limits for calculating probabilities")
    accuracy.setValue(7.5)
    widgets["accuracy"] = accuracy

    max_lost_frames = QtWidgets.QSpinBox()
    max_lost_frames.setName("max lost")
    max_lost_frames.setToolTip(
        "Number of frames without observation before marking as lost"
    )
    max_lost_frames.setValue(5)
    widgets["max_lost"] = max_lost_frames

    return widgets
