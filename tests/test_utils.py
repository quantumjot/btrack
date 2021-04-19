import numpy as np
import pytest

from btrack import btypes, utils


def _make_test_image(size=128, ndim=2, nobj=10, binary=True):
    shape = (size,) * ndim
    img = np.zeros(shape, dtype=np.uint16)
    if nobj == 0:
        return img, None
    centroids = np.random.choice(128, size=(ndim, nobj), replace=False)
    vals = 1 if binary else 1 + np.arange(nobj)
    img[tuple(centroids.tolist())] = vals

    # sort the centroids by axis
    centroids = np.transpose(centroids)
    centroids = centroids[
        np.lexsort([centroids[:, dim] for dim in range(ndim)][::-1])
    ]

    return img, centroids


def _example_segmentation_generator():
    for i in range(10):
        img, centroids = _make_test_image()
        yield img


def _validate_centroids(centroids, objects, scale=None):
    """Take a list of objects and validate them agains the ground truth."""

    if centroids is None:
        assert not objects
        return

    if scale is not None:
        centroids = centroids * np.array(scale)

    obj_as_array = np.array([[obj.z, obj.y, obj.x] for obj in objects])
    if centroids.shape[-1] == 2:
        obj_as_array = obj_as_array[:, 1:]

    np.testing.assert_equal(obj_as_array, centroids)


def test_segmentation_to_objects_type():
    """Test that btrack objects are returned."""
    img, centroids = _make_test_image()
    objects = utils.segmentation_to_objects(img[np.newaxis, ...])
    assert all([isinstance(o, btypes.PyTrackObject) for o in objects])


def test_segmentation_to_objects_type_generator():
    """Test generator as input."""
    generator = _example_segmentation_generator()
    objects = utils.segmentation_to_objects(generator)
    assert all([isinstance(o, btypes.PyTrackObject) for o in objects])


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("nobj", [0, 1, 10, 30])
@pytest.mark.parametrize("binary", [True, False])
def test_segmentation_to_objects(ndim, nobj, binary):
    """Test different types of segmentation images."""
    img, centroids = _make_test_image(ndim=ndim, nobj=nobj, binary=True)
    objects = utils.segmentation_to_objects(img[np.newaxis, ...])
    _validate_centroids(centroids, objects)


@pytest.mark.parametrize("scale", [None, (1.0, 1.0), (1.0, 10.0), (10.0, 1.0)])
def test_segmentation_to_objects_scale(scale):
    """Test anisotropic scaling."""
    img, centroids = _make_test_image()
    objects = utils.segmentation_to_objects(img[np.newaxis, ...], scale=scale)
    _validate_centroids(centroids, objects, scale)
