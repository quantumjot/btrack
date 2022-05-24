import h5py
import numpy as np
import pytest

from btrack import btypes, utils
from btrack.dataio import HDF5FileHandler


def _make_test_image(
    boxsize: int = 150,
    ndim: int = 2,
    nobj: int = 10,
    binsize: int = 5,
    binary: bool = True,
):
    """Make a test image that ensures that no two pixels are in contact."""
    shape = (boxsize,) * ndim
    img = np.zeros(shape, dtype=np.uint16)

    # return an empty image if we have no objects
    if nobj == 0:
        return img, None

    # split this into voxels
    bins = boxsize // binsize

    def _sample():
        _img = np.zeros((binsize,) * ndim, dtype=np.uint16)
        _coord = tuple(
            np.random.randint(1, binsize - 1, size=(ndim,)).tolist()
        )
        _img[_coord] = 1
        assert np.sum(_img) == 1
        return _img, _coord

    # now we update nobj grid positions with a sample
    grid = np.stack(np.meshgrid(*[np.arange(bins)] * ndim), -1).reshape(
        -1, ndim
    )
    rng = np.random.default_rng(seed=1234)
    rbins = rng.choice(grid, size=(nobj,), replace=False)

    # iterate over the bins and add a smaple
    centroids = []
    for v, bin in enumerate(rbins):
        sample, point = _sample()
        slices = tuple(
            [slice(b * binsize, b * binsize + binsize, 1) for b in bin]
        )
        val = 1 if binary else v + 1
        img[slices] = sample * val

        # shift the actual coordinates back to image space
        point = point + bin * binsize  # - 0.5
        centroids.append(point)

    # sort the centroids by axis
    centroids_sorted = np.array(centroids)
    centroids_sorted = centroids_sorted[
        np.lexsort([centroids_sorted[:, dim] for dim in range(ndim)][::-1])
    ]

    assert centroids_sorted.shape[0] == nobj

    vals = np.unique(img)
    assert np.max(vals) == 1 if binary else nobj
    return img, centroids_sorted


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
    img, centroids = _make_test_image()
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
    img, centroids = _make_test_image(ndim=ndim, nobj=nobj, binary=binary)
    objects = utils.segmentation_to_objects(img[np.newaxis, ...])
    _validate_centroids(centroids, objects)


def test_dask_segmentation_to_objects():
    """Test using a dask array as segmentation input."""
    img, centroids = _make_test_image()
    da = pytest.importorskip(
        "dask.array", reason="Dask not installed in pytest environment."
    )
    img = da.from_array(img)
    objects = utils.segmentation_to_objects(img[np.newaxis, ...])
    _validate_centroids(centroids, objects)


@pytest.mark.parametrize("scale", [None, (1.0, 1.0), (1.0, 10.0), (10.0, 1.0)])
def test_segmentation_to_objects_scale(scale):
    """Test anisotropic scaling."""
    img, centroids = _make_test_image()
    objects = utils.segmentation_to_objects(img[np.newaxis, ...], scale=scale)
    _validate_centroids(centroids, objects, scale)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("nobj", [0, 1, 10, 30, 300])
def test_assign_class_ID(ndim, nobj):
    """Test mask class_id assignment."""
    img, centroids = _make_test_image(ndim=ndim, nobj=nobj, binary=False)
    objects = utils.segmentation_to_objects(
        img[np.newaxis, ...], assign_class_ID=True
    )
    # check that the values match
    for obj in objects:
        centroid = (int(obj.z), int(obj.y), int(obj.x))[-ndim:]
        assert obj.properties["class_id"] == img[centroid]


def test_regionprops():
    """Test using regionprops returns objects with correct property keys."""
    img, centroids = _make_test_image()
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
    img, centroids = _make_test_image(ndim=ndim, binary=True)
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


def _load_segmentation_and_tracks():
    f = h5py.File("./tests/_test_data/update_segmentation_data.h5", "r")
    coords = tuple(f[c][:] for c in ["tc", "yc", "xc"])
    in_segmentation = np.zeros((10, 1024, 1020), dtype=f["in_values"].dtype)
    out_segmentation = np.zeros((10, 1024, 1020), dtype=f["out_values"].dtype)
    in_segmentation[coords] = f["in_values"][:]
    out_segmentation[coords] = f["out_values"][:]
    tracks = HDF5FileHandler("./tests/_test_data/tracks.h5").tracks
    return in_segmentation, out_segmentation, tracks


def test_update_segmentation_2d():
    in_segmentation, out_segmentation, tracks = _load_segmentation_and_tracks()
    relabeled = utils.update_segmentation(in_segmentation, tracks)
    assert np.allclose(relabeled, out_segmentation)


def test_update_segmentation_3d():
    in_segmentation, out_segmentation, tracks = _load_segmentation_and_tracks()

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
