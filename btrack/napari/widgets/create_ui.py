from __future__ import annotations

from qtpy import QtWidgets

from napari.viewer import Viewer

from btrack.napari.widgets._general import (
    create_config_widgets,
    create_input_widgets,
    create_logo_widgets,
    create_track_widgets,
    create_update_method_widgets,
)
from btrack.napari.widgets._motion import create_motion_model_widgets
from btrack.napari.widgets._optimiser import create_optimiser_widgets


class BtrackWidget(QtWidgets.QScrollArea):
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

        # Let the scroll area automatically resize the widget
        self.setWidgetResizable(True)  # noqa: FBT003

        self._main_layout = QtWidgets.QVBoxLayout()
        self._main_widget = QtWidgets.QWidget()
        self._main_widget.setLayout(self._main_layout)
        self.setWidget(self._main_widget)
        self._tabs = QtWidgets.QTabWidget()

        # Create widgets and add to layout
        self._widgets = {}

        self._add_logo_widgets()
        self._add_input_widgets()
        # This must be added after the input widgets
        self._main_layout.addWidget(self._tabs, stretch=0)
        self._add_update_method_widgets()
        self._add_motion_model_widgets()
        self._add_optimiser_widgets()
        self._add_config_widgets()
        self._add_track_widgets()

        # Expand the main widget
        self._main_layout.addStretch(stretch=1)

        # Add attribute access for each widget
        for name, widget in self._widgets.items():
            self.__setattr__(
                name,
                widget,
            )

    def _add_logo_widgets(self) -> None:
        """Adds the btrack logo with a link to the documentation"""
        logo_widgets = create_logo_widgets()
        self._widgets.update(logo_widgets)
        for widget in logo_widgets.values():
            self._main_layout.addWidget(widget, stretch=0)

    def _add_input_widgets(self) -> None:
        """Create input widgets and add to main layout"""
        labels_and_widgets = create_input_widgets()
        self._widgets.update(
            {key: value[1] for key, value in labels_and_widgets.items()}
        )

        widget_holder = QtWidgets.QGroupBox("Input")
        layout = QtWidgets.QFormLayout()
        for label, widget in labels_and_widgets.values():
            label_widget = QtWidgets.QLabel(label)
            label_widget.setToolTip(widget.toolTip())
            layout.addRow(label_widget, widget)
        widget_holder.setLayout(layout)
        self._main_layout.addWidget(widget_holder, stretch=0)

    def _add_update_method_widgets(self) -> None:
        """Create update method widgets and add to main layout"""
        labels_and_widgets = create_update_method_widgets()
        self._widgets.update(
            {key: value[1] for key, value in labels_and_widgets.items()}
        )

        layout = QtWidgets.QFormLayout()
        for label, widget in labels_and_widgets.values():
            label_widget = QtWidgets.QLabel(label)
            label_widget.setToolTip(widget.toolTip())
            layout.addRow(label_widget, widget)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)
        self._tabs.addTab(tab, "Method")

    def _add_motion_model_widgets(self) -> None:
        """Create motion model widgets and add to main layout"""
        labels_and_widgets = create_motion_model_widgets()
        self._widgets.update(
            {key: value[1] for key, value in labels_and_widgets.items()}
        )

        layout = QtWidgets.QFormLayout()
        for label, widget in labels_and_widgets.values():
            label_widget = QtWidgets.QLabel(label)
            label_widget.setToolTip(widget.toolTip())
            layout.addRow(label_widget, widget)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)
        self._tabs.addTab(tab, "Motion")

    def _add_optimiser_widgets(self) -> None:
        """Create hypothesis model widgets and add to main layout"""
        labels_and_widgets = create_optimiser_widgets()
        self._widgets.update(
            {key: value[1] for key, value in labels_and_widgets.items()}
        )

        layout = QtWidgets.QFormLayout()
        for label, widget in labels_and_widgets.values():
            label_widget = QtWidgets.QLabel(label)
            label_widget.setToolTip(widget.toolTip())
            layout.addRow(label_widget, widget)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)
        self._tabs.addTab(tab, "Optimiser")

    def _add_config_widgets(self) -> None:
        """Creates the IO widgets related to the user config"""
        io_widgets = create_config_widgets()
        self._widgets.update(io_widgets)

        layout = QtWidgets.QFormLayout()
        for widget in io_widgets.values():
            layout.addRow(widget)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)
        self._tabs.addTab(tab, "Config")

    def _add_track_widgets(self) -> None:
        """Create widgets for running the tracking"""
        track_widgets = create_track_widgets()
        self._widgets.update(track_widgets)
        for widget in track_widgets.values():
            self._main_layout.addWidget(widget, stretch=0)
