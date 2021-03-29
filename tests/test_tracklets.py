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


def _create_test_properties():
    properties = {
        "speed": np.random.uniform(0.0, 1.0),
        "circularity": np.random.uniform(0.0, 1.0),
        "reporter": np.random.uniform(0.0, 1.0),
    }
    return properties


def _create_test_tracklet(track_len: int):
    """Create a test track."""
    data = [_create_test_object()[0] for i in range(track_len)]
    props = [_create_test_properties() for i in range(track_len)]
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


def test_object():
    """Test that an object is correctly instantiated, and that the stored
    data matches the data used for creation."""
    obj, data = _create_test_object()
    for k, v in data.items():
        assert getattr(obj, k) == v


@pytest.mark.parametrize("properties", [{}, _create_test_properties()])
def test_object_properties(properties: dict):
    """Test an object with some properties."""
    obj, data = _create_test_object()
    obj.properties = properties
    for k, v in properties.items():
        assert obj.properties[k] == v


@pytest.mark.parametrize("track_len", [0, 1, 10, 100, 1000])
def test_tracklet(track_len: int):
    """Test that a track is correctly instantiated, and that the stored
    data matches the data used for creation."""
    tracklet, data, properties, track_ID = _create_test_tracklet(track_len)
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
    tracklet, data, properties, track_ID = _create_test_tracklet(track_len)
    t_properties = tracklet.properties

    for k, v in properties.items():
        np.testing.assert_equal(t_properties[k], v)


# def test_malformed_tracklet_properties():
#     """Test for malformed properties by truncating the first property."""
#     track_len = 10
#     tracklet, data, properties, track_ID = _create_test_tracklet(track_len)
#     first_key = list(properties.keys())[0]
#     # this removes a property from one of the objects
#     del tracklet._data[0].properties[first_key]
#
#     # raises a key error when trying to retreive the properties
#     with pytest.raises(KeyError):
#         _ = tracklet.properties
