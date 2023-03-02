import numpy as np
import pytest

from tests._utils import (
    create_test_object,
    create_test_properties,
    create_test_tracklet,
)


def test_create_object():
    """Test that an object is correctly instantiated, and that the stored
    data matches the data used for creation."""
    obj, data = create_test_object()
    for k, v in data.items():
        np.testing.assert_equal(getattr(obj, k), v)


@pytest.mark.parametrize("properties", [{}, create_test_properties()])
def test_create_object_with_properties(properties: dict):
    """Test an object with some properties."""
    obj, data = create_test_object()
    obj.properties = properties
    for k, v in properties.items():
        np.testing.assert_equal(obj.properties[k], v)


@pytest.mark.parametrize("properties", [{}, create_test_properties()])
def test_object_features(properties: dict):
    """Test creating object and setting tracking features."""
    obj, _ = create_test_object()
    obj.properties = properties
    assert obj.n_features == 0
    keys = list(properties.keys())
    obj.set_features(keys)
    n_keys = sum([np.asarray(p).size for p in properties.values()])
    assert obj.n_features == n_keys


def test_object_feature_values():
    """Test creating object with features and returning correct values."""
    obj, _ = create_test_object()
    properties = create_test_properties()
    obj.properties = properties
    keys = list(properties.keys())
    obj.set_features(keys)

    # check that the numpy data matches
    raw_features = np.ctypeslib.as_array(obj.features, shape=(obj.n_features,))
    flat_properties = np.concatenate(
        [np.asarray(obj.properties[k]).ravel() for k in keys], axis=0
    )
    np.testing.assert_almost_equal(flat_properties, raw_features)


def test_object_incorrect_features():
    """Test creating object and setting tracking features."""
    obj, data = create_test_object()
    assert obj.n_features == 0
    with pytest.raises(KeyError):
        obj.set_features(
            [
                "this_key_does_not_exist",
            ]
        )
    assert obj.n_features == 0


@pytest.mark.parametrize("track_len", [0, 1, 10, 100, 1000])
def test_create_tracklet(track_len: int):
    """Test that a track is correctly instantiated, and that the stored
    data matches the data used for creation."""
    tracklet, data, properties, track_ID = create_test_tracklet(track_len)
    assert len(tracklet) == len(data)

    # now check that the track data is correct
    assert track_ID == tracklet.ID
    fields = ["x", "y", "z", "t"]
    for field in fields:
        obj_data = [getattr(obj, field) for obj in data]
        np.testing.assert_equal(obj_data, getattr(tracklet, field))


@pytest.mark.parametrize("track_len", [0, 1, 10, 100, 1000])
def test_create_tracklet_with_properties(track_len: int):
    """Test that a track is correctly instantiated, and that the stored
    properties match the data used for creation."""
    tracklet, data, properties, track_ID = create_test_tracklet(track_len)
    t_properties = tracklet.properties

    for k, v in properties.items():
        np.testing.assert_equal(t_properties[k], v)
