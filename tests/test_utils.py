from btrack import btypes, utils
from btrack.constants import DEFAULT_OBJECT_KEYS, Dimensionality
from btrack.io import objects_from_array
from btrack.io._localization import _is_unique

from ._utils import create_test_image, create_test_tracklet

from contextlib import nullcontext

import numpy as np
import pytest


def test_is_unique_boolean_array():
    """Test that _is_unique returns False for boolean arrays."""
    # Create a simple boolean mask
    mask = np.array(
        [[True, False, False], [False, True, False], [False, False, True]], dtype=bool
    )
    assert _is_unique(mask) is False


def test_is_unique_labeled_array():
    """Test that _is_unique returns True for already-labeled arrays."""
    # Create a labeled array (each object has a unique integer ID)
    labeled = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=int)
    assert _is_unique(labeled) is True


def test_is_unique_unlabeled_array():
    """Test that _is_unique returns False for unlabeled binary arrays."""
    # Create a mask with multiple separate objects all having the same
    # non-sequential ID (5). After labeling, they should have IDs (1, 2)
    unlabeled = np.array(
        [[5, 5, 0, 0], [5, 5, 0, 0], [0, 0, 5, 5], [0, 0, 5, 5]], dtype=int
    )
    assert _is_unique(unlabeled) is False


def _example_segmentation_generator():
    for _ in range(10):
        img, _centroids = create_test_image()
        yield img


def _validate_centroids(centroids, objects, scale=None):
    """Take a list of objects and validate them agains the ground truth."""

    if centroids is None:
        assert not objects
        return

    if scale is not None:
        centroids = centroids * np.array(scale)

    ndim = centroids.shape[-1]

    obj_as_array = np.array([[obj.z, obj.y, obj.x] for obj in objects])
    if ndim == Dimensionality.TWO:
        obj_as_array = obj_as_array[:, 1:]

    # sort the centroids by axis
    centroids = centroids[np.lexsort([centroids[:, dim] for dim in range(ndim)][::-1])]

    # sort the objects
    obj_as_array = obj_as_array[
        np.lexsort([obj_as_array[:, dim] for dim in range(ndim)][::-1])
    ]

    np.testing.assert_equal(obj_as_array, centroids)


def test_segmentation_to_objects_type():
    """Test that btrack objects are returned."""
    img, _centroids = create_test_image()
    objects = utils.segmentation_to_objects(img[np.newaxis, ...])
    assert all(isinstance(o, btypes.PyTrackObject) for o in objects)


def test_segmentation_to_objects_type_generator():
    """Test generator as input."""
    generator = _example_segmentation_generator()
    objects = utils.segmentation_to_objects(generator)
    assert all(isinstance(o, btypes.PyTrackObject) for o in objects)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("nobj", [0, 1, 10, 30, 300])
@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("num_workers", [1, 4])
def test_segmentation_to_objects(ndim, nobj, binary, num_workers):
    """Test different types of segmentation images."""
    img, centroids = create_test_image(ndim=ndim, nobj=nobj, binary=binary)
    objects = utils.segmentation_to_objects(
        img[np.newaxis, ...], num_workers=num_workers
    )
    _validate_centroids(centroids, objects)


def test_dask_segmentation_to_objects():
    """Test using a dask array as segmentation input."""
    img, centroids = create_test_image()
    da = pytest.importorskip(
        "dask.array", reason="Dask not installed in pytest environment."
    )
    img = da.from_array(img)
    objects = utils.segmentation_to_objects(img[np.newaxis, ...])
    _validate_centroids(centroids, objects)


@pytest.mark.parametrize("scale", [None, (1.0, 1.0), (1.0, 10.0), (10.0, 1.0)])
def test_segmentation_to_objects_scale(scale):
    """Test anisotropic scaling."""
    img, centroids = create_test_image()
    objects = utils.segmentation_to_objects(img[np.newaxis, ...], scale=scale)
    _validate_centroids(centroids, objects, scale)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("nobj", [0, 1, 10, 30, 300])
def test_assign_class_ID(ndim, nobj):
    """Test mask class_id assignment."""
    img, _centroids = create_test_image(ndim=ndim, nobj=nobj, binary=False)
    objects = utils.segmentation_to_objects(img[np.newaxis, ...], assign_class_ID=True)
    # check that the values match
    for obj in objects:
        centroid = (int(obj.z), int(obj.y), int(obj.x))[-ndim:]
        assert obj.properties["class_id"] == img[centroid]


def test_regionprops():
    """Test using regionprops returns objects with correct property keys."""
    img, _centroids = create_test_image()
    properties = (
        "area",
        "axis_major_length",
    )
    objects = utils.segmentation_to_objects(img[np.newaxis, ...], properties=properties)

    # check that the properties keys match
    for obj in objects:
        assert set(obj.properties.keys()) == set(properties)


def test_extra_regionprops():
    """Test adding a callable function for extra property calculation."""
    img, _centroids = create_test_image()

    def extra_prop(_mask) -> float:
        return np.sum(_mask)

    extra_properties = (extra_prop,)

    objects = utils.segmentation_to_objects(
        img[np.newaxis, ...],
        extra_properties=extra_properties,
    )

    extra_prop_keys = [fn.__name__ for fn in extra_properties]

    # check that the properties keys match
    for obj in objects:
        assert set(obj.properties.keys()) == set(extra_prop_keys)


@pytest.mark.parametrize("ndim", [2, 3])
def test_intensity_image(default_rng, ndim):
    """Test using an intensity image."""
    img, _centroids = create_test_image(ndim=ndim, binary=True)
    intensity_image = img * default_rng.uniform(size=img.shape)
    objects = utils.segmentation_to_objects(
        img[np.newaxis, ...],
        intensity_image=intensity_image[np.newaxis, ...],
        use_weighted_centroid=True,
        properties=("max_intensity",),
    )
    # check that the values match
    for obj in objects:
        centroid = (int(obj.z), int(obj.y), int(obj.x))[-ndim:]
        assert obj.properties["max_intensity"] == intensity_image[centroid]


def test_update_segmentation_2d(test_segmentation_and_tracks):
    """Test relabeling a 2D-segmentation with track ID."""
    in_segmentation, out_segmentation, tracks = test_segmentation_and_tracks
    relabeled = utils.update_segmentation(in_segmentation, tracks)
    assert np.allclose(relabeled, out_segmentation)


@pytest.mark.parametrize("color_by", ["ID", "root", "generation", "fake"])
def test_update_segmentation_2d_colorby(test_segmentation_and_tracks, color_by):
    """Test relabeling a 2D-segmentation with track ID."""
    in_segmentation, _out_segmentation, tracks = test_segmentation_and_tracks

    with pytest.raises(ValueError) if color_by == "fake" else nullcontext():
        _ = utils.update_segmentation(in_segmentation, tracks, color_by=color_by)


def test_update_segmentation_3d(test_segmentation_and_tracks):
    """Test relabeling a 3D-segmentation with track ID."""
    in_segmentation, out_segmentation, tracks = test_segmentation_and_tracks

    in_segmentation = np.broadcast_to(
        in_segmentation[:, None],
        (in_segmentation.shape[0], 5, *in_segmentation.shape[-2:]),
    )

    out_segmentation = np.broadcast_to(
        out_segmentation[:, None],
        (in_segmentation.shape[0], 5, *in_segmentation.shape[-2:]),
    )

    relabeled = utils.update_segmentation(in_segmentation, tracks)
    assert np.allclose(relabeled, out_segmentation)


@pytest.mark.parametrize("ndim", [2, 3])
def test_tracks_to_napari(ndim: int):
    """Test converting from `btrack` Tracklets to a `napari` compatible data
    structure."""

    # make three fake tracks with properties
    track_len = 10
    tracks = [create_test_tracklet(track_len, idx + 1)[0] for idx in range(3)]

    # set up a fake graph
    tracks[0].children = [2, 3]
    tracks[1].parent = 1
    tracks[2].parent = 1

    data, properties, graph = utils.tracks_to_napari(tracks, ndim=ndim)

    # check the data is of the correct shape (ID, T + ndim)
    assert data.shape[-1] == ndim + 2

    # check that the data have the correct values
    track_ids = np.asarray([1] * track_len + [2] * track_len + [3] * track_len)
    np.testing.assert_equal(data[:, 0], track_ids)
    header = ["t", *["z", "y", "x"][-ndim:]]
    for idx, key in enumerate(header):
        gt_data = np.concatenate([getattr(t, key) for t in tracks])
        np.testing.assert_equal(data[:, idx + 1], gt_data)

    # check the graph
    assert graph == {2: [1], 3: [1]}

    # check the properties keys are correct, note that nD keys are replaced with
    # keys that start with the property key, e.g. `nD` is replaced with `nD-0`
    # and so forth
    for key in tracks[0].properties:
        assert any(k.startswith(key) for k in properties)


@pytest.mark.parametrize("ndim", [1, 4])
def test_tracks_to_napari_incorrect_ndim(ndim: int):
    """Test that providing incorrect dimensions to `tracks_to_napari` raises a
    `ValueError`."""
    # make three fake tracks with properties
    track_len = 10
    tracks = [create_test_tracklet(track_len, idx + 1)[0] for idx in range(3)]

    with pytest.raises(ValueError):
        _data, _properties, _graph = utils.tracks_to_napari(tracks, ndim=ndim)


@pytest.mark.parametrize("ndim", [2, 3])
def test_tracks_to_napari_ndim_inference(ndim: int):
    """Test inferring the correct dimensions from track data when using
    `tracks_to_napari`."""

    # make a fake track with n dimensions
    track_len = 10
    tracks = [create_test_tracklet(track_len, 1, ndim=ndim)[0]]
    data, _, _ = utils.tracks_to_napari(tracks, ndim=None)

    # check the data is of the correct shape (ID, T + ndim)
    assert data.shape[-1] == ndim + 2


def test_napari_to_tracks(sample_tracks):
    """Test that a napari Tracks layer can be converted to a list of Tracklets.

    First convert tracks to a napari layer, then convert back and compare.
    """

    data, properties, graph = utils.tracks_to_napari(sample_tracks)
    tracks = utils.napari_to_tracks(data, properties, graph)

    properties_to_compare = [
        "ID",
        "t",
        "x",
        "y",
        # "z",  # z-coordinates are different
        "parent",
        "label",
        "state",
        "root",
        "is_root",
        "is_leaf",
        "start",
        "stop",
        "generation",
        "dummy",
        "properties",
    ]

    sample_tracks_dicts = [
        sample.to_dict(properties_to_compare) for sample in sample_tracks
    ]
    tracks_dicts = [track.to_dict(properties_to_compare) for track in tracks]
    assert sample_tracks_dicts == tracks_dicts


def test_objects_from_array(test_objects):
    """Test creation of a list of objects from a numpy array."""

    obj_arr = np.stack(
        [[getattr(obj, k) for k in DEFAULT_OBJECT_KEYS] for obj in test_objects],
        axis=0,
    )

    obj_from_arr = objects_from_array(obj_arr)

    assert obj_arr.shape[0] == len(test_objects)
    assert obj_arr.shape[-1] == len(DEFAULT_OBJECT_KEYS)

    assert len(obj_from_arr) == len(test_objects)

    for test_obj, obj in zip(test_objects, obj_from_arr):
        assert isinstance(obj, btypes.PyTrackObject)
        for key in DEFAULT_OBJECT_KEYS:
            assert getattr(test_obj, key) == getattr(obj, key)
