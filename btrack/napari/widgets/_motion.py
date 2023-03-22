from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magicgui.widgets import Widget

import magicgui


def _make_label_bold(label: str) -> str:
    """Generate html for a bold label"""

    return f"<b>{label}</b>"


def _create_sigma_widgets() -> list[Widget]:
    """Create widgets for setting the magnitudes of the MotionModel matrices"""

    P_sigma_tooltip = (
        "Magnitude of error in initial estimates.\n"
        "Used to scale the matrix P."
    )
    P_sigma = magicgui.widgets.create_widget(
        value=150.0,
        name="P_sigma",
        label=f"max({_make_label_bold('P')})",
        widget_type="FloatSpinBox",
        options={"tooltip": P_sigma_tooltip},
    )

    G_sigma_tooltip = (
        "Magnitude of error in process.\n Used to scale the matrix G."
    )
    G_sigma = magicgui.widgets.create_widget(
        value=15.0,
        name="G_sigma",
        label=f"max({_make_label_bold('G')})",
        widget_type="FloatSpinBox",
        options={"tooltip": G_sigma_tooltip},
    )

    R_sigma_tooltip = (
        "Magnitude of error in measurements.\n Used to scale the matrix R."
    )
    R_sigma = magicgui.widgets.create_widget(
        value=5.0,
        name="R_sigma",
        label=f"max({_make_label_bold('R')})",
        widget_type="FloatSpinBox",
        options={"tooltip": R_sigma_tooltip},
    )

    return [
        P_sigma,
        G_sigma,
        R_sigma,
    ]


def create_motion_model_widgets() -> list[Widget]:
    """Create widgets for setting parameters of the MotionModel"""

    motion_model_label = magicgui.widgets.create_widget(
        label=_make_label_bold("Motion model"),
        widget_type="Label",
        gui_only=True,
    )

    sigma_widgets = _create_sigma_widgets()

    accuracy_tooltip = "Integration limits for calculating probabilities"
    accuracy = magicgui.widgets.create_widget(
        value=7.5,
        name="accuracy",
        label="accuracy",
        widget_type="FloatSpinBox",
        options={"tooltip": accuracy_tooltip},
    )

    max_lost_frames_tooltip = (
        "Number of frames without observation before marking as lost"
    )
    max_lost_frames = magicgui.widgets.create_widget(
        value=5,
        name="max_lost",
        label="max lost",
        widget_type="SpinBox",
        options={"tooltip": max_lost_frames_tooltip},
    )

    return [
        motion_model_label,
        *sigma_widgets,
        accuracy,
        max_lost_frames,
    ]
