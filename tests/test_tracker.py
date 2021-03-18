import json

import numpy as np

import btrack


def _load_csv():
    objects = btrack.dataio.import_CSV("./tests/_test_data/test_data.csv")
    return objects


def _load_ground_truth():
    with open("./tests/_test_data/test_ground_truth.json", "r") as file:
        ground_truth = json.load(file)
    return ground_truth


def test_tracker():
    """Test the operation of the tracker, using the default config and known
    data."""
    objects = _load_csv()
    ground_truth = _load_ground_truth()

    # run the tracking
    with btrack.BayesianTracker() as tracker:
        tracker.configure_from_file("./models/cell_config.json")
        tracker.append(objects)
        tracker.volume = ((0, 1600), (0, 1200), (-1e5, 1e5))
        tracker.track_interactive(step_size=100)
        tracker.optimize()
        tracks = tracker.tracks

    # iterate over the tracks and check that the object references match
    for track in tracks:
        gt_refs = ground_truth[str(track.ID)]
        np.testing.assert_equal(track.refs, gt_refs)
