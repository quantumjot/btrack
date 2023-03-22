from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magicgui.widgets import Widget

from btrack.napari.widgets._general import (
    create_control_widgets,
    create_input_widgets,
    create_update_method_widgets,
)
from btrack.napari.widgets._hypothesis import create_hypothesis_model_widgets
from btrack.napari.widgets._motion import create_motion_model_widgets


def create_widgets() -> list[Widget]:
    """Create all the widgets for the plugin"""

    input_widgets = create_input_widgets()
    update_method_widgets = create_update_method_widgets()
    motion_model_widgets = create_motion_model_widgets()
    hypothesis_model_widgets = create_hypothesis_model_widgets()
    control_buttons = create_control_widgets()

    return [
        *input_widgets,
        *update_method_widgets,
        *motion_model_widgets,
        *hypothesis_model_widgets,
        *control_buttons,
    ]
