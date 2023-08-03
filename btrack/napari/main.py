from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt
    from qtpy import QtWidgets

    from btrack.config import TrackerConfig
    from btrack.napari.config import TrackerConfigs

import logging

import napari

import btrack
import btrack.napari.config
import btrack.napari.sync
import btrack.napari.widgets
from btrack.utils import segmentation_to_objects

__all__ = [
    "create_btrack_widget",
]

# get the logger instance
logger = logging.getLogger(__name__)

# if we don't have any handlers, set one up
if not logger.handlers:
    # configure stream handler
    log_fmt = logging.Formatter(
        "[%(levelname)s][%(asctime)s] %(message)s",
        datefmt="%Y/%m/%d %I:%M:%S %p",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_fmt)

    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)


def create_btrack_widget() -> btrack.napari.widgets.BtrackWidget:
    """Create widgets for the btrack plugin."""

    # First create our UI along with some default configs for the widgets
    all_configs = btrack.napari.config.create_default_configs()
    btrack_widget = btrack.napari.widgets.BtrackWidget(
        napari_viewer=napari.current_viewer(),
    )

    # Set the cell_config defaults in the gui
    btrack.napari.sync.update_widgets_from_config(
        unscaled_config=all_configs["cell"],
        btrack_widget=btrack_widget,
    )

    # Add any existing Labels layers to the segmentation selector
    add_existing_labels(
        viewer=btrack_widget.viewer,
        combobox=btrack_widget.segmentation,
    )

    # Now set the callbacks
    btrack_widget.viewer.layers.events.inserted.connect(
        lambda event: select_inserted_labels(
            new_layer=event.value,
            combobox=btrack_widget.segmentation,
        ),
    )

    btrack_widget.viewer.layers.events.removed.connect(
        lambda event: remove_deleted_labels(
            deleted_layer=event.value,
            combobox=btrack_widget.segmentation,
        ),
    )

    btrack_widget.config.currentTextChanged.connect(
        lambda selected: select_config(btrack_widget, all_configs, selected),
    )

    btrack_widget.call_button.clicked.connect(
        lambda: run(btrack_widget, all_configs),
    )

    btrack_widget.reset_button.clicked.connect(
        lambda: restore_defaults(btrack_widget, all_configs),
    )

    btrack_widget.save_config_button.clicked.connect(
        lambda: save_config_to_json(btrack_widget, all_configs)
    )

    btrack_widget.load_config_button.clicked.connect(
        lambda: load_config_from_json(btrack_widget, all_configs)
    )

    return btrack_widget


def add_existing_labels(
    viewer: napari.Viewer,
    combobox: QtWidgets.QComboBox,
):
    """Add all existing Labels layers in the viewer to a combobox"""

    labels_layers = [
        layer.name
        for layer in viewer.layers
        if isinstance(layer, napari.layers.Labels)
    ]
    combobox.addItems(labels_layers)


def select_inserted_labels(
    new_layer: napari.layers.Layer,
    combobox: QtWidgets.QComboBox,
):
    """Update the selected Labels when a labels layer is added"""

    if not isinstance(new_layer, napari.layers.Labels):
        message = (
            f"Not selecting new layer {new_layer.name} as input for the "
            f"segmentation widget as {new_layer.name} is {type(new_layer)} "
            "layer not an Labels layer."
        )
        logger.debug(message)
        return

    combobox.addItem(new_layer.name)
    combobox.setCurrentText(new_layer.name)

    # Update layer name when it changes
    viewer = napari.current_viewer()
    new_layer.events.name.connect(
        lambda event: update_labels_name(
            layer=event.source,
            labels_layers=[
                layer
                for layer in viewer.layers
                if isinstance(layer, napari.layers.Labels)
            ],
            combobox=combobox,
        ),
    )


def update_labels_name(
    layer: napari.layers.Layer,
    labels_layers: list[napari.layer.Layer],
    combobox: QtWidgets.QComboBox,
):
    """Update the name of an Labels layer"""

    if not isinstance(layer, napari.layers.Labels):
        message = (
            f"Not updating name of layer {layer.name} as input for the "
            f"segmentation widget as {layer.name} is {type(layer)} "
            "layer not a Labels layer."
        )
        logger.debug(message)
        return

    layer_index = [layer.name for layer in labels_layers].index(layer.name)
    combobox.setItemText(layer_index, layer.name)


def remove_deleted_labels(
    deleted_layer: napari.layers.Layer,
    combobox: QtWidgets.QComboBox,
):
    """Remove the deleted Labels layer name from the combobox"""

    if not isinstance(deleted_layer, napari.layers.Labels):
        message = (
            f"Not deleting layer {deleted_layer.name} from the segmentation "
            f"widget as {deleted_layer.name} is {type(deleted_layer)} "
            "layer not an Labels layer."
        )
        logger.debug(message)
        return

    layer_index = combobox.findText(deleted_layer.name)
    combobox.removeItem(layer_index)


def select_config(
    btrack_widget: btrack.napari.widgets.BtrackWidget,
    configs: TrackerConfigs,
    new_config_name: str,
) -> None:
    """Set widget values from a newly-selected base config"""

    # first update the previous config with the current widget values
    previous_config_name = configs.current_config
    previous_config = configs[previous_config_name]
    previous_config = btrack.napari.sync.update_config_from_widgets(
        unscaled_config=previous_config,
        btrack_widget=btrack_widget,
    )

    # now load the newly-selected config and set widget values
    configs.current_config = new_config_name
    new_config = configs[new_config_name]
    new_config = btrack.napari.sync.update_widgets_from_config(
        unscaled_config=new_config,
        btrack_widget=btrack_widget,
    )


def run(
    btrack_widget: btrack.napari.widgets.BtrackWidget,
    configs: TrackerConfigs,
) -> None:
    """
    Update the TrackerConfig from widget values, run tracking,
    and add tracks to the viewer.
    """

    # TODO:
    # This method of showing the activity dock will be removed
    # and replaced with a public method in the api
    # See: https://github.com/napari/napari/issues/4598
    activity_dock_visible = (
        btrack_widget.viewer.window._qt_window._activity_dialog.isVisible()
    )
    btrack_widget.viewer.window._status_bar._toggle_activity_dock(visible=True)

    if btrack_widget.segmentation.currentIndex() < 0:
        napari.utils.notifications.show_error(
            "No segmentation (Image layer) selected - cannot run tracking."
        )
        return

    unscaled_config = configs[btrack_widget.config.currentText()]
    unscaled_config = btrack.napari.sync.update_config_from_widgets(
        unscaled_config=unscaled_config,
        btrack_widget=btrack_widget,
    )

    config = unscaled_config.scale_config()
    segmentation_name = btrack_widget.segmentation.currentText()
    segmentation = btrack_widget.viewer.layers[segmentation_name]
    data, properties, graph = _run_tracker(segmentation, config)

    btrack_widget.viewer.add_tracks(
        data=data,
        properties=properties,
        graph=graph,
        name=f"{segmentation}_btrack",
        scale=segmentation.scale,
        translate=segmentation.translate,
    )

    btrack_widget.viewer.window._status_bar._toggle_activity_dock(
        activity_dock_visible
    )

    message = f"Finished tracking for '{segmentation_name}'"
    napari.utils.notifications.show_info(message)


def _run_tracker(
    segmentation: napari.layers.Image | napari.layers.Labels,
    tracker_config: TrackerConfig,
) -> tuple[npt.NDArray, dict, dict]:
    """
    Runs BayesianTracker with given segmentation and configuration.
    """
    with btrack.BayesianTracker() as tracker, napari.utils.progress(
        total=5
    ) as pbr:
        pbr.set_description("Initialising the tracker")
        tracker.configure(tracker_config)
        pbr.update(1)

        # append the objects to be tracked
        pbr.set_description("Convert segmentation to trackable objects")
        segmented_objects = segmentation_to_objects(segmentation.data)
        pbr.update(1)
        tracker.append(segmented_objects)

        # set the volume
        # btrack order of dimensions is XY(Z)
        # napari order of dimensions is T(Z)XY
        # so we ignore the first dimension (time) and reverse the others
        dimensions = segmentation.level_shapes[0, 1:]
        tracker.volume = tuple(
            (0, dimension) for dimension in reversed(dimensions)
        )

        # track them (in interactive mode)
        pbr.set_description("Run tracking")
        tracker.track(step_size=100)
        pbr.update(1)

        # generate hypotheses and run the global optimizer
        pbr.set_description("Run optimisation")
        tracker.optimize()
        pbr.update(1)

        # get the tracks in a format for napari visualization
        pbr.set_description("Convert to napari tracks layer")
        data, properties, graph = tracker.to_napari()
        pbr.update(1)

        return data, properties, graph


def restore_defaults(
    btrack_widget: btrack.napari.widgets.BtrackWidget,
    configs: TrackerConfigs,
) -> None:
    """Reload the config file then set widgets to the config's default values."""

    config_name = configs.current_config
    filename = configs[config_name].filename
    configs.add_config(
        filename=filename,
        overwrite=True,
        name=config_name,
    )

    config = configs[config_name]
    config = btrack.napari.sync.update_widgets_from_config(
        unscaled_config=config,
        btrack_widget=btrack_widget,
    )


def save_config_to_json(
    btrack_widget: btrack.napari.widgets.BtrackWidget,
    configs: TrackerConfigs,
) -> None:
    """Save widget values to file"""

    save_path = btrack.napari.widgets.save_path_dialogue_box()
    if save_path is None:
        _msg = (
            "btrack napari plugin: Configuration not saved - "
            "operation cancelled by the user."
        )
        logger.info(_msg)
        return

    unscaled_config = configs[btrack_widget.config.currentText()]
    btrack.napari.sync.update_config_from_widgets(
        unscaled_config=unscaled_config,
        btrack_widget=btrack_widget,
    )
    config = unscaled_config.scale_config()

    btrack.config.save_config(save_path, config)


def load_config_from_json(
    btrack_widget: btrack.napari.widgets.BtrackWidget,
    configs: TrackerConfigs,
) -> None:
    """Load a config from file and set it as the selected base config"""

    load_path = btrack.napari.widgets.load_path_dialogue_box()
    if load_path is None:
        _msg = "btrack napari plugin: No file loaded - operation cancelled by the user."
        logger.info(_msg)
        return

    config_name = configs.add_config(filename=load_path, overwrite=False)
    btrack_widget.config.addItem(config_name)
    btrack_widget.config.setCurrentText(config_name)
