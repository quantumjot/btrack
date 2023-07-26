from __future__ import annotations

from qtpy import QtWidgets

from napari.viewer import Viewer

from btrack.napari.widgets._general import (
    create_control_widgets,
    create_input_widgets,
    create_update_method_widgets,
)
from btrack.napari.widgets._hypothesis import create_hypothesis_model_widgets
from btrack.napari.widgets._motion import create_motion_model_widgets


def create_widgets() -> dict[str, QtWidgets.QWidget]:
    """Create all the widgets for the plugin"""

    input_widgets = create_input_widgets()
    update_method_widgets = create_update_method_widgets()
    motion_model_widgets = create_motion_model_widgets()
    hypothesis_model_widgets = create_hypothesis_model_widgets()
    control_buttons = create_control_widgets()

    return {
        **input_widgets,
        **update_method_widgets,
        **motion_model_widgets,
        **hypothesis_model_widgets,
        **control_buttons,
    }


class BtrackWidget(QtWidgets.QWidget):
    """Main btrack widget"""

    def __init__(self, napari_viewer: Viewer) -> None:
        """Instansiates the primary widget in napari.

        Args:
            napari_viewer: A napari viewer instance
        """
        super().__init__()

        self._viewer = napari_viewer
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        # Create widgets and add to layout
        self._add_input_widgets()
        self._add_update_method_widgets()
        self._add_motion_model_widgets()
        self._add_hypothesis_model_widgets()
        self._add_control_buttons_widgets()
        self._widgets = {
            **self._input_widgets,
            **self._update_method_widgets,
            **self._motion_model_widgets,
            **self._hypothesis_model_widgets,
            **self._control_buttons,
        }

    def _add_input_widgets(self):
        """Create input widgets and add to main layout"""
        self._input_widgets = create_input_widgets()
        self._widgets.update(self._input_widgets)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()
        for label, widget in self._input_widgets:
            layout.addRow(label, widget)
        widget.setLayout(layout)
        self._layout.addWidget(widget)

    def _add_update_method_widgets(self):
        """Create update method widgets and add to main layout"""
        self._update_method_widgets = create_update_method_widgets()
        self._widgets.update(self._update_method_widgets)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()
        for label, widget in self._update_method_widgets:
            layout.addRow(label, widget)
        widget.setLayout(layout)
        self._layout.addWidget(widget)

    def _add_motion_model_widgets(self):
        """Create motion model widgets and add to main layout"""
        self._motion_model_widgets = create_motion_model_widgets()
        self._widgets.update(self._motion_model_widgets)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()
        for label, widget in self._motion_model_widgets:
            layout.addRow(label, widget)
        widget.setLayout(layout)
        self._layout.addWidget(widget)

    def _add_hypothesis_model_widgets(self):
        """Create hypothesis model widgets and add to main layout"""
        self._hypothesis_model_widgets = create_hypothesis_model_widgets()
        self._widgets.update(self._hypothesis_model_widgets)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()
        for label, widget in self._hypothesis_model_widgets:
            layout.addRow(label, widget)
        widget.setLayout(layout)
        self._layout.addWidget(widget)

    def _add_control_buttons_widgets(self):
        """Create control buttons widgets and add to main layout"""
        self._control_buttons_widgets = create_control_widgets()
        self._widgets.update(self._control_buttons_widgets)

        for widget in self._control_buttons_widgets.values():
            self._layout.addWidget(widget)
