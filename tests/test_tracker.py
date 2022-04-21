import json

import numpy as np
from _utils import full_tracker_example, simple_tracker_example

import btrack


def _gt_object_hook(d):
    """JSON stores keys as strings, convert these to integers."""
    return {int(k): v for k, v in d.items()}


def _load_csv():
    objects = btrack.dataio.import_CSV("./tests/_test_data/test_data.csv")
    return objects


def _load_ground_truth():
    with open("./tests/_test_data/test_ground_truth.json", "r") as file:
        ground_truth = json.load(file, object_hook=_gt_object_hook)
    return ground_truth


def _load_ground_truth_graph():
    with open("./tests/_test_data/test_graph.json", "r") as file:
        ground_truth_graph = json.load(file, object_hook=_gt_object_hook)

    return ground_truth_graph


def _get_tracklet(tracks: list, idx: int) -> list:
    """Get a tracklet by the first object ID"""
    target = [t for t in tracks.values() if t[0] == idx]
    if target:
        return target[0]
    else:
        raise ValueError("Object ID not found.")


def test_tracker():
    """Test the tracks output of the tracker, using the default config and known
    data."""
    objects = _load_csv()
    ground_truth = _load_ground_truth()

    tracker = full_tracker_example(objects)
    tracks = tracker.tracks

    # iterate over the tracks and check that the object references match
    for track in tracks:
        gt_refs = _get_tracklet(ground_truth, track.refs[0])
        np.testing.assert_equal(track.refs, gt_refs)


def test_tracker_graph():
    """Test the graph output of the tracker, using the default config and known
    data."""

    objects = _load_csv()
    ground_truth_graph = _load_ground_truth_graph()

    # run the tracking
    tracker = full_tracker_example(objects)
    _, _, graph = tracker.to_napari(ndim=2)

    assert ground_truth_graph == graph


def test_tracker_frames():
    """Test to make sure all frames are accounted for."""

    tracker, objects = simple_tracker_example()
    tracks = tracker.tracks

    assert len(tracks) == 1
    track = tracks[0]
    np.testing.assert_equal(track.t, objects["t"])
