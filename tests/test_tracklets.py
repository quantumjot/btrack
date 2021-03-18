import numpy as np
import pytest

import btrack


def _create_test_object():
    """Create a test object."""
    data = {
        "ID": np.random.randint(0, 1000),
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


def _create_test_tracklet(track_len: int):
    """Create a test track."""
    data = [_create_test_object()[0] for i in range(track_len)]
    track_ID = np.random.randint(0, 1000)
    tracklet = btrack.btypes.Tracklet(track_ID, data)
    return tracklet, data, track_ID


def _create_test_properties(track_len: int):
    properties = {
        "speed": np.random.uniform(0.0, 1.0, size=(track_len,)),
        "circularity": np.random.uniform(0.0, 1.0, size=(track_len,)),
        "reporter": np.random.uniform(0.0, 1.0, size=(track_len,)),
    }
    return properties


def test_object():
    """Test that an object is correctly instantiated, and that the stored
    data matches the data used for creation."""
    obj, data = _create_test_object()
    for k, v in data.items():
        assert getattr(obj, k) == v


@pytest.mark.parametrize("track_len", [0, 1, 10, 100, 1000])
def test_tracklet(track_len: int):
    """Test that a track is correctly instantiated, and that the stored
    data matches the data used for creation."""
    tracklet, data, track_ID = _create_test_tracklet(track_len)
    assert len(tracklet) == len(data)

    # now check that the track data is correct
    assert tracklet.ID == track_ID
    fields = ["x", "y", "z", "t"]
    for field in fields:
        obj_data = [getattr(obj, field) for obj in data]
        np.testing.assert_equal(obj_data, getattr(tracklet, field))


@pytest.mark.parametrize("track_len", [0, 1, 10, 100, 1000])
def test_tracklet_properties(track_len: int):
    """Test that a track is correctly instantiated, and that the stored
    properties match the data used for creation."""
    tracklet, data, track_ID = _create_test_tracklet(track_len)
    properties = _create_test_properties(track_len)

    tracklet.properties = properties
    t_properties = tracklet.properties

    for k, v in properties.items():
        np.testing.assert_equal(t_properties[k], v)


def test_malformed_tracklet_properties():
    """Test for malformed properties by truncating the first property."""
    track_len = 10
    tracklet, data, track_ID = _create_test_tracklet(track_len)
    properties = _create_test_properties(track_len)
    first_key = list(properties.keys())[0]
    properties[first_key] = properties[first_key][: track_len - 2]

    with pytest.raises(ValueError):
        tracklet.properties = properties
