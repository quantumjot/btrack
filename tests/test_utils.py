import numpy as np
import pytest

from btrack import btypes, utils

from ._utils import create_test_image, create_test_tracklet


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
def test_intensity_image(default_rng, ndim):
    """Test using an intensity image."""
    img, centroids = create_test_image(ndim=ndim, binary=True)
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
    tracks = [create_test_tracklet(10, idx + 1)[0] for idx in range(3)]

    # set up a fake graph
    tracks[0].children = [2, 3]
    tracks[1].parent = 1
    tracks[2].parent = 1

    data, properties, graph = utils.tracks_to_napari(tracks, ndim=ndim)

    # check the data
    if ndim == 2:
        assert data.shape[-1] == 4
    else:
        assert data.shape[-1] == 5

    # check the graph
    assert graph == {2: [1], 3: [1]}
