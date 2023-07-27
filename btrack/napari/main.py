from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from btrack.config import TrackerConfig
    from btrack.napari.config import TrackerConfigs

import logging

from qtpy import QtWidgets

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


def create_btrack_widget() -> QtWidgets.QWidget:
    """Create widgets for the btrack plugin."""

    # First create our UI along with some default configs for the widgets
    all_configs = btrack.napari.config.create_default_configs()
    btrack_widget = btrack.napari.widgets.BtrackWidget(
        napari_viewer=napari.current_viewer(),
    )

    # Set the cell_config defaults in the gui
    btrack.napari.sync.update_widgets_from_config(
        unscaled_config=all_configs["cell"],
        container=btrack_widget,
    )

    # Now set the callbacks
    btrack_widget._viewer.layers.events.inserted.connect(
        lambda event: select_inserted_image(
            new_layer=event.value,
            widget=btrack_widget.segmentation,
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

    # there are lots of widgets so make the container scrollable
    # scroll = QtWidgets.QScrollArea()
    # scroll.setWidget(btrack_widget._widget._qwidget)
    # btrack_widget._widget._qwidget = scroll

    return btrack_widget


def select_inserted_image(
    new_layer: napari.layers.Layer,
    widget: QtWidgets.QComboBox,
):
    """Update the selected Image when a image layer is added"""

    if not isinstance(new_layer, napari.layers.Image):
        message = (
            f"Not selecting new layer {new_layer.name} as input for the "
            f"segmentation widget as {new_layer.name} is {type(new_layer)} "
            "layer not an Image layer."
        )
        logger.debug(message)
        return

    widget.addItem(new_layer.name)
    widget.setCurrentText(new_layer.name)


# TODO: automatically update layer name in QComboBox when it is changed in the viewer
def update_image_name(
    layer: napari.layers.Layer,
    new_name: str,
    widget: QtWidgets.QComboBox,
):
    """Update the name of an Image layer"""

    if not isinstance(layer, napari.layers.Image):
        message = (
            f"Not selecting new layer {layer.name} as input for the "
            f"segmentation widget as {layer.name} is {type(layer)} "
            "layer not an Image layer."
        )
        logger.debug(message)
        return


def select_config(
    btrack_widget: QtWidgets.QWidget,
    configs: TrackerConfigs,
    new_config_name: str,
) -> None:
    """Set widget values from a newly-selected base config"""

    # first update the previous config with the current widget values
    previous_config_name = configs.current_config
    previous_config = configs[previous_config_name]
    previous_config = btrack.napari.sync.update_config_from_widgets(
        unscaled_config=previous_config,
        container=btrack_widget,
    )

    # now load the newly-selected config and set widget values
    configs.current_config = new_config_name
    new_config = configs[new_config_name]
    new_config = btrack.napari.sync.update_widgets_from_config(
        unscaled_config=new_config,
        container=btrack_widget,
    )


def run(btrack_widget: QtWidgets.QWidget, configs: TrackerConfigs) -> None:
    """
    Update the TrackerConfig from widget values, run tracking,
    and add tracks to the viewer.
    """

    if btrack_widget.segmentation.currentIndex() < 0:
        napari.utils.notifications.show_error(
            "No segmentation (Image layer) selected - cannot run tracking."
        )
        return

    unscaled_config = configs[btrack_widget.config.currentText()]
    unscaled_config = btrack.napari.sync.update_config_from_widgets(
        unscaled_config=unscaled_config,
        container=btrack_widget,
    )

    config = unscaled_config.scale_config()
    segmentation_name = btrack_widget.segmentation.currentText()
    segmentation = btrack_widget._viewer.layers[segmentation_name]
    data, properties, graph = _run_tracker(segmentation, config)

    btrack_widget._viewer.add_tracks(
        data=data,
        properties=properties,
        graph=graph,
        name=f"{segmentation}_btrack",
        scale=segmentation.scale,
        translate=segmentation.translate,
    )


def _run_tracker(
    segmentation: napari.layers.Image | napari.layers.Labels,
    tracker_config: TrackerConfig,
) -> tuple[npt.NDArray, dict, dict]:
    """
    Runs BayesianTracker with given segmentation and configuration.
    """
    with btrack.BayesianTracker() as tracker:
        tracker.configure(tracker_config)

        # append the objects to be tracked
        segmented_objects = segmentation_to_objects(segmentation.data)
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
        tracker.track_interactive(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks in a format for napari visualization
        data, properties, graph = tracker.to_napari()
        return data, properties, graph


def restore_defaults(
    btrack_widget: QtWidgets.QWidget, configs: TrackerConfigs
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
        container=btrack_widget,
    )


def save_config_to_json(
    btrack_widget: QtWidgets.QWidget, configs: TrackerConfigs
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

    unscaled_config = configs[btrack_widget.config.current_choice]
    btrack.napari.sync.update_config_from_widgets(
        unscaled_config=unscaled_config,
        container=btrack_widget,
    )
    config = unscaled_config.scale_config()

    btrack.config.save_config(save_path, config)


def load_config_from_json(
    btrack_widget: QtWidgets.QWidget, configs: TrackerConfigs
) -> None:
    """Load a config from file and set it as the selected base config"""

    load_path = btrack.napari.widgets.load_path_dialogue_box()
    if load_path is None:
        _msg = "btrack napari plugin: No file loaded - operation cancelled by the user."
        logger.info(_msg)
        return

    config_name = configs.add_config(filename=load_path, overwrite=False)
    btrack_widget.config.options["choices"].append(config_name)
    btrack_widget.config.reset_choices()
    btrack_widget.config.value = config_name
