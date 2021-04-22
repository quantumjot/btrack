import numpy as np

import btrack


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
