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

    return (
        create_input_widgets()
        | create_update_method_widgets()
        | create_motion_model_widgets()
        | create_hypothesis_model_widgets()
        | create_control_widgets()
    )


class BtrackWidget(QtWidgets.QWidget):
    """Main btrack widget"""

    def __getitem__(self, key: str) -> QtWidgets.QWidgets:
        return self._widgets[key]

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
        self._widgets = {}
        self._add_input_widgets()
        self._add_update_method_widgets()
        self._add_motion_model_widgets()
        self._add_hypothesis_model_widgets()
        self._add_control_buttons_widgets()
        for name, widget in self._widgets.items():
            self.__setattr__(
                name,
                widget,
            )

    def _add_input_widgets(self):
        """Create input widgets and add to main layout"""
        labels_and_widgets = create_input_widgets()
        self._input_widgets = {
            key: value[1] for key, value in labels_and_widgets.items()
        }
        self._widgets.update(self._input_widgets)

        widget_holder = QtWidgets.QGroupBox("Input")
        layout = QtWidgets.QFormLayout()
        for label, widget in labels_and_widgets.values():
            layout.addRow(QtWidgets.QLabel(label), widget)
        widget_holder.setLayout(layout)
        self._layout.addWidget(widget_holder)

    def _add_update_method_widgets(self):
        """Create update method widgets and add to main layout"""
        labels_and_widgets = create_update_method_widgets()
        self._update_method_widgets = {
            key: value[1] for key, value in labels_and_widgets.items()
        }
        self._widgets.update(self._update_method_widgets)

        widget_holder = QtWidgets.QGroupBox("Method")
        layout = QtWidgets.QFormLayout()
        for label, widget in labels_and_widgets.values():
            layout.addRow(QtWidgets.QLabel(label), widget)
        widget_holder.setLayout(layout)
        self._layout.addWidget(widget_holder)

    def _add_motion_model_widgets(self):
        """Create motion model widgets and add to main layout"""
        labels_and_widgets = create_motion_model_widgets()
        self._motion_model_widgets = {
            key: value[1] for key, value in labels_and_widgets.items()
        }
        self._widgets.update(self._motion_model_widgets)

        widget_holder = QtWidgets.QGroupBox("Motion")
        layout = QtWidgets.QFormLayout()
        for label, widget in labels_and_widgets.values():
            layout.addRow(QtWidgets.QLabel(label), widget)
        widget_holder.setLayout(layout)
        self._layout.addWidget(widget_holder)

    def _add_hypothesis_model_widgets(self):
        """Create hypothesis model widgets and add to main layout"""
        labels_and_widgets = create_hypothesis_model_widgets()
        self._hypothesis_model_widgets = {
            key: value[1] for key, value in labels_and_widgets.items()
        }
        self._widgets.update(self._hypothesis_model_widgets)

        widget_holder = QtWidgets.QGroupBox("Hypothesis")
        layout = QtWidgets.QFormLayout()
        for label, widget in labels_and_widgets.values():
            layout.addRow(QtWidgets.QLabel(label), widget)
        widget_holder.setLayout(layout)
        self._layout.addWidget(widget_holder)

    def _add_control_buttons_widgets(self):
        """Create control buttons widgets and add to main layout"""
        self._control_buttons_widgets = create_control_widgets()
        self._widgets.update(self._control_buttons_widgets)

        for widget in self._control_buttons_widgets.values():
            self._layout.addWidget(widget)
