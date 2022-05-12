from pathlib import Path
from random import choices
from typing import Optional

from traitlets import default

import btrack
import napari
from btrack.utils import segmentation_to_objects
from magicgui import magicgui
from magicgui.widgets import FunctionGui

from btrack.config import load_config
from btrack import datasets

default_config = load_config(datasets.cell_config())

hypothesis_model = default_config.hypothesis_model
hypothesis_model_widgets = {}
for parameter, default_value in hypothesis_model:
    hypothesis_model_widgets[parameter]=dict(value=default_value)

def run_tracker(objects, config_file_path):
    with btrack.BayesianTracker() as tracker:
        # configure the tracker using a config file
        tracker.configure_from_file(config_file_path)
        tracker.max_search_radius = 50

        # append the objects to be tracked
        tracker.append(objects)

        # set the volume
        tracker.volume = ((0, 1600), (0, 1200), (-1e5, 64.0))

        # track them (in interactive mode)
        tracker.track_interactive(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks in a format for napari visualization
        data, properties, graph = tracker.to_napari(ndim=2)
        return data, properties, graph


def track() -> FunctionGui:
    @magicgui(
        call_button=True,
        persist=True,
        dt=dict(value=default_config.motion_model.dt, step=0.01),
        **hypothesis_model_widgets,
        reset_button=dict(widget_type="PushButton", text="Reset defaults"),
    )
    def widget(
        viewer: napari.Viewer,
        segmentation: napari.layers.Image,
        dt: float,
        name: str,
        hypotheses: str,
        lambda_time: float,
        lambda_dist: float,
        lambda_link: float,
        lambda_branch: float,
        eta: float,
        theta_dist: float,
        theta_time: float,
        dist_thresh: int,
        time_thresh: int,
        apop_thresh: int,
        segmentation_miss_rate: float,
        apoptosis_rate: float,
        relax: bool,
        reset_button,
    ):
        segmented_objects = segmentation_to_objects(segmentation.data[:100, ...])
        data, properties, graph = run_tracker(segmented_objects, datasets.cell_config())
        viewer.add_tracks(
            data=data, properties=properties, graph=graph, name=f"{segmentation}_btrack"
        )

    return widget
