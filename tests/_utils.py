from pathlib import Path

import numpy as np

import btrack

CONFIG_FILE = Path("./models/cell_config.json")


def create_test_object(id=None):
    """Create a test object."""
    data = {
        "ID": np.random.randint(0, 1000) if id is None else int(id),
        "x": np.random.uniform(0.0, 1000.0),
        "y": np.random.uniform(0.0, 1000.0),
        "z": np.random.uniform(0.0, 1000.0),
        "t": np.random.randint(0, 1000),
        "dummy": False,
        "states": 5,
        "label": 0,
        "prob": 0.5,
    }

    obj = btrack.btypes.PyTrackObject().from_dict(data)
    return obj, data


def create_test_properties():
    properties = {
        "speed": np.random.uniform(0.0, 1.0),
        "circularity": np.random.uniform(0.0, 1.0),
        "reporter": np.random.uniform(0.0, 1.0),
    }
    return properties


def create_test_tracklet(track_len: int):
    """Create a test track."""
    data = [create_test_object()[0] for i in range(track_len)]
    props = [create_test_properties() for i in range(track_len)]
    for idx, obj in enumerate(data):
        obj.properties = props[idx]
    track_ID = np.random.randint(0, 1000)
    tracklet = btrack.btypes.Tracklet(track_ID, data)

    # convert to dictionary {key: [p0,...,pn]}
    if not props:
        properties = {}
    else:
        properties = {k: [p[k] for p in props] for k in props[0].keys()}

    return tracklet, data, properties, track_ID


def full_tracker_example(objects):
    # run the tracking
    tracker = btrack.BayesianTracker()
    tracker.configure(CONFIG_FILE)
    tracker.append(objects)
    tracker.volume = ((0, 1600), (0, 1200), (-1e5, 1e5))
    tracker.track_interactive(step_size=100)
    tracker.optimize()
    return tracker


def simple_tracker_example():
    """Run a simple tracker example with some data."""
    x = np.array([200, 201, 202, 203, 204, 207, 208])
    y = np.array([503, 507, 499, 500, 510, 515, 518])
    t = np.array([0, 1, 2, 3, 4, 5, 6])
    z = np.zeros(x.shape)

    objects_dict = {"x": x, "y": y, "z": z, "t": t}
    objects = btrack.dataio.objects_from_dict(objects_dict)

    tracker = full_tracker_example(objects)
    return tracker, objects_dict
