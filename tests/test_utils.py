import numpy as np
import pytest

from btrack import btypes, utils

from ._utils import create_test_image, create_test_segmentation_and_tracks


def _example_segmentation_generator():
    for i in range(10):
        img, centroids = create_test_image()
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
    if ndim == 2:
        obj_as_array = obj_as_array[:, 1:]

    # sort the centroids by axis
    centroids = centroids[
        np.lexsort([centroids[:, dim] for dim in range(ndim)][::-1])
    ]

    # sort the objects
    obj_as_array = obj_as_array[
        np.lexsort([obj_as_array[:, dim] for dim in range(ndim)][::-1])
    ]

    np.testing.assert_equal(obj_as_array, centroids)


def test_segmentation_to_objects_type():
    """Test that btrack objects are returned."""
    img, centroids = create_test_image()
    objects = utils.segmentation_to_objects(img[np.newaxis, ...])
    assert all([isinstance(o, btypes.PyTrackObject) for o in objects])


def test_segmentation_to_objects_type_generator():
    """Test generator as input."""
    generator = _example_segmentation_generator()
    objects = utils.segmentation_to_objects(generator)
    assert all([isinstance(o, btypes.PyTrackObject) for o in objects])


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("nobj", [0, 1, 10, 30, 300])
@pytest.mark.parametrize("binary", [True, False])
def test_segmentation_to_objects(ndim, nobj, binary):
    """Test different types of segmentation images."""
    img, centroids = create_test_image(ndim=ndim, nobj=nobj, binary=binary)
    objects = utils.segmentation_to_objects(img[np.newaxis, ...])
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
    img, centroids = create_test_image(ndim=ndim, nobj=nobj, binary=False)
    objects = utils.segmentation_to_objects(
        img[np.newaxis, ...], assign_class_ID=True
    )
    # check that the values match
    for obj in objects:
        centroid = (int(obj.z), int(obj.y), int(obj.x))[-ndim:]
        assert obj.properties["class_id"] == img[centroid]


def test_regionprops():
    """Test using regionprops returns objects with correct property keys."""
    img, centroids = create_test_image()
    properties = (
        "area",
        "axis_major_length",
    )
    objects = utils.segmentation_to_objects(
        img[np.newaxis, ...], properties=properties
    )

    # check that the properties keys match
    for obj in objects:
        assert set(obj.properties.keys()) == set(properties)


@pytest.mark.parametrize("ndim", [2, 3])
def test_intensity_image(ndim):
    """Test using an intensity image."""
    img, centroids = create_test_image(ndim=ndim, binary=True)
    rng = np.random.default_rng(seed=1234)
    intensity_image = img * rng.uniform(size=img.shape)
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


def test_segmentation_tracks():
    seg, tracks = create_test_segmentation_and_tracks(ndim=2, binary=False)
    assert seg.shape == (10, 128, 128)
