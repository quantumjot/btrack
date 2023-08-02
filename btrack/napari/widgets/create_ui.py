from __future__ import annotations

from qtpy import QtWidgets

from napari.viewer import Viewer

from btrack.napari.widgets._general import (
    create_input_widgets,
    create_io_widgets,
    create_update_method_widgets,
)
from btrack.napari.widgets._hypothesis import create_hypothesis_model_widgets
from btrack.napari.widgets._motion import create_motion_model_widgets


def create_widgets() -> (
    dict[str, QtWidgets.QWidget | tuple(str, QtWidgets.QWidget)]
):
    """Create all the widgets for the plugin"""

    return (
        create_input_widgets()
        | create_update_method_widgets()
        | create_motion_model_widgets()
        | create_hypothesis_model_widgets()
        | create_io_widgets()
    )


class BtrackWidget(QtWidgets.QWidget):
    """Main btrack widget"""

    def __getitem__(self, key: str) -> QtWidgets.QWidget:
        return self._widgets[key]

    def __init__(self, napari_viewer: Viewer) -> None:
        """Instantiates the primary widget in napari.

        Args:
            napari_viewer: A napari viewer instance
        """
        super().__init__()

        # We will need to viewer for various callbacks
        self.viewer = napari_viewer

        self._layout = QtWidgets.QVBoxLayout(self)
        self._tabs = QtWidgets.QTabWidget()

        # Create widgets and add to layout
        self._widgets = {}
        self._add_input_widgets()
        self._add_update_method_widgets()
        self._add_motion_model_widgets()
        self._add_hypothesis_model_widgets()
        self._add_io_widgets()
        for name, widget in self._widgets.items():
            self.__setattr__(
                name,
                widget,
            )
        self._layout.addWidget(self._tabs)

    def _add_input_widgets(self) -> None:
        """Create input widgets and add to main layout"""
        labels_and_widgets = create_input_widgets()
        self._widgets.update(
            {key: value[1] for key, value in labels_and_widgets.items()}
        )

        widget_holder = QtWidgets.QGroupBox("Input")
        layout = QtWidgets.QFormLayout()
        for label, widget in labels_and_widgets.values():
            layout.addRow(QtWidgets.QLabel(label), widget)
        widget_holder.setLayout(layout)
        self._layout.addWidget(widget_holder)

    def _add_update_method_widgets(self) -> None:
        """Create update method widgets and add to main layout"""
        labels_and_widgets = create_update_method_widgets()
        self._widgets.update(
            {key: value[1] for key, value in labels_and_widgets.items()}
        )

        tab = QtWidgets.QScrollArea()
        self._tabs.addTab(tab, "Method")
        widget_holder = QtWidgets.QWidget()
        tab.setWidget(widget_holder)
        # Let the scroll area automatically resize the widget
        tab.setWidgetResizable(True)  # noqa: FBT003

        layout = QtWidgets.QFormLayout()
        for label, widget in labels_and_widgets.values():
            layout.addRow(QtWidgets.QLabel(label), widget)
        widget_holder.setLayout(layout)

    def _add_motion_model_widgets(self) -> None:
        """Create motion model widgets and add to main layout"""
        labels_and_widgets = create_motion_model_widgets()
        self._widgets.update(
            {key: value[1] for key, value in labels_and_widgets.items()}
        )

        tab = QtWidgets.QScrollArea()
        self._tabs.addTab(tab, "Motion")
        widget_holder = QtWidgets.QWidget()
        tab.setWidget(widget_holder)
        # Let the scroll area automatically resize the widget
        tab.setWidgetResizable(True)  # noqa: FBT003

        layout = QtWidgets.QFormLayout()
        for label, widget in labels_and_widgets.values():
            layout.addRow(QtWidgets.QLabel(label), widget)
        widget_holder.setLayout(layout)

    def _add_hypothesis_model_widgets(self) -> None:
        """Create hypothesis model widgets and add to main layout"""
        labels_and_widgets = create_hypothesis_model_widgets()
        self._widgets.update(
            {key: value[1] for key, value in labels_and_widgets.items()}
        )

        tab = QtWidgets.QScrollArea()
        self._tabs.addTab(tab, "Hypothesis")
        widget_holder = QtWidgets.QWidget()
        tab.setWidget(widget_holder)
        # Let the scroll area automatically resize the widget
        tab.setWidgetResizable(True)  # noqa: FBT003

        layout = QtWidgets.QFormLayout()
        for label, widget in labels_and_widgets.values():
            layout.addRow(QtWidgets.QLabel(label), widget)
        widget_holder.setLayout(layout)

    def _add_io_widgets(self) -> None:
        """Creates the IO widgets related to the user config"""
        io_widgets = create_io_widgets()
        self._widgets.update(io_widgets)

        tab = QtWidgets.QScrollArea()
        self._tabs.addTab(tab, "I/O")
        widget_holder = QtWidgets.QWidget()
        tab.setWidget(widget_holder)
        # Let the scroll area automatically resize the widget
        tab.setWidgetResizable(True)  # noqa: FBT003

        layout = QtWidgets.QFormLayout()
        for widget in io_widgets.values():
            layout.addRow(widget)
        widget_holder.setLayout(layout)
