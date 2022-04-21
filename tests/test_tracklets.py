import numpy as np
import pytest
from ._utils import (
    create_test_object,
    create_test_properties,
    create_test_tracklet,
)


def test_object():
    """Test that an object is correctly instantiated, and that the stored
    data matches the data used for creation."""
    obj, data = create_test_object()
    for k, v in data.items():
        assert getattr(obj, k) == v


@pytest.mark.parametrize("properties", [{}, create_test_properties()])
def test_object_properties(properties: dict):
    """Test an object with some properties."""
    obj, data = create_test_object()
    obj.properties = properties
    for k, v in properties.items():
        assert obj.properties[k] == v


@pytest.mark.parametrize("track_len", [0, 1, 10, 100, 1000])
def test_tracklet(track_len: int):
    """Test that a track is correctly instantiated, and that the stored
    data matches the data used for creation."""
    tracklet, data, properties, track_ID = create_test_tracklet(track_len)
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
    tracklet, data, properties, track_ID = create_test_tracklet(track_len)
    t_properties = tracklet.properties

    for k, v in properties.items():
        np.testing.assert_equal(t_properties[k], v)
