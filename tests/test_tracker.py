from btrack import BayesianTracker

from ._utils import (
    TEST_DATA_PATH,
    full_tracker_example,
    simple_tracker_example,
)

import json
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        pytest.param((), {}, None, id="configured-defaults"),
        pytest.param(
            ({"tm_lim": 2500},), {}, {"tm_lim": 2500}, id="positional-options"
        ),
        pytest.param(
            (), {"options": {"tm_lim": 2500}}, {"tm_lim": 2500}, id="named-options"
        ),
        pytest.param((), {"options": {}}, {}, id="empty-options"),
        pytest.param((), {"tm_lim": 2500}, {"tm_lim": 2500}, id="legacy-keywords"),
        pytest.param(
            ({"tm_lim": 2500, "msg_lev": 0},),
            {"tm_lim": 1000},
            {"tm_lim": 1000, "msg_lev": 0},
            id="keyword-overrides",
        ),
    ],
)
def test_optimize_forwards_glpk_options(monkeypatch, args, kwargs, expected):
    """Check that the alias passes a flat options dictionary to the optimizer."""
    with BayesianTracker() as tracker:
        optimise = Mock(return_value=[])
        monkeypatch.setattr(tracker, "optimise", optimise)

        assert tracker.optimize(*args, **kwargs) is optimise.return_value

    optimise.assert_called_once_with(options=expected)


def _gt_object_hook(d):
    """JSON stores keys as strings, convert these to integers."""
    return {int(k): v for k, v in d.items()}


def _load_ground_truth():
    with open(TEST_DATA_PATH / "test_ground_truth.json", "r") as file:
        ground_truth = json.load(file, object_hook=_gt_object_hook)
    return ground_truth


def _load_ground_truth_graph() -> dict:
    with open(TEST_DATA_PATH / "test_graph.json", "r") as file:
        ground_truth_graph = json.load(file, object_hook=_gt_object_hook)

    return ground_truth_graph


def _get_tracklet(tracks: dict, idx: int) -> list:
    """Get a tracklet by the first object ID"""
    if target := [t for t in tracks.values() if t[0] == idx]:
        return target[0]
    else:
        raise ValueError("Object ID not found.")


def test_tracker(test_real_objects):
    """Test the tracks output of the tracker, using the default config and known
    data."""
    ground_truth = _load_ground_truth()

    tracker = full_tracker_example(test_real_objects)
    tracks = tracker.tracks

    # iterate over the tracks and check that the object references match
    for track in tracks:
        gt_refs = _get_tracklet(ground_truth, track.refs[0])
        np.testing.assert_equal(track.refs, gt_refs)


def test_tracker_kalman(test_real_objects):
    """Test the Kalman filter output of the tracker."""
    tracker = full_tracker_example(
        test_real_objects,
    )
    tracker.return_kalman = True
    assert tracker.return_kalman

    tracks = tracker.tracks

    for track in tracks:
        assert track.kalman.shape[0] == len(track)
        assert track.mu(0).shape == (3, 1)
        assert track.covar(0).shape == (3, 3)
        assert track.predicted(0).shape == (3, 1)

        np.testing.assert_equal(
            np.array([track.x[0], track.y[0], track.z[0]]),
            np.squeeze(track.mu(0)),
        )


def test_tracker_graph(test_real_objects):
    """Test the graph output of the tracker, using the default config and known
    data."""

    ground_truth_graph = _load_ground_truth_graph()

    # run the tracking
    tracker = full_tracker_example(test_real_objects)
    _, _, graph = tracker.to_napari(ndim=2)

    assert ground_truth_graph == graph


def test_tracker_frames():
    """Test to make sure all frames are accounted for."""

    tracker, objects = simple_tracker_example()
    tracks = tracker.tracks

    assert len(tracks) == 1
    track = tracks[0]
    np.testing.assert_equal(track.t, objects["t"])


@pytest.mark.parametrize("store_graph", [False, True])
def test_tracker_candidate_graph(test_real_objects, store_graph):
    """Check that a candidate graph is returned if we set the flag to store it."""
    tracker = full_tracker_example(
        test_real_objects,
        store_candidate_graph=store_graph,
    )
    assert tracker.store_candidate_graph == store_graph
    edges = tracker.candidate_graph_edges()

    # graph should contain edges
    assert bool(edges) == store_graph, f"Found {len(edges)} edges in candidate graph."
