from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt
from skimage.measure import label

import btrack

CONFIG_FILE = Path(__file__).resolve().parent.parent / "models" / "cell_config.json"

TEST_DATA_PATH = Path(__file__).resolve().parent / "_test_data"

RANDOM_SEED = 1234


def create_test_object(
    test_id: int | None = None,
) -> tuple[btrack.btypes.PyTrackObject, dict[str, Any]]:
    """Create a test object."""

    rng = np.random.default_rng(seed=RANDOM_SEED)

    data = {
        "ID": rng.integers(0, 1000) if test_id is None else int(test_id),
        "x": rng.uniform(0.0, 1000.0),
        "y": rng.uniform(0.0, 1000.0),
        "z": rng.uniform(0.0, 1000.0),
        "t": rng.integers(0, 1000),
        "dummy": False,
        "label": 0,
    }

    obj = btrack.btypes.PyTrackObject().from_dict(data)
    return obj, data


def create_test_properties() -> dict[str, float]:
    """Create test properties for an object."""
    rng = np.random.default_rng(seed=RANDOM_SEED)
    return {
        "speed": rng.uniform(0.0, 1.0),
        "circularity": rng.uniform(0.0, 1.0),
        "reporter": rng.uniform(0.0, 1.0),
        "nD": rng.uniform(0.0, 1.0, size=(5,)),
    }


def create_test_tracklet(
    track_len: int,
    track_id: int | None = None,
) -> tuple[
    btrack.btypes.Tracklet,
    list[btrack.btypes.PyTrackObject],
    list[dict[str, Any]],
    int,
]:
    """Create a test track."""
    rng = np.random.default_rng(seed=RANDOM_SEED)

    data = [create_test_object()[0] for _ in range(track_len)]
    props = [create_test_properties() for _ in range(track_len)]
    for idx, obj in enumerate(data):
        obj.properties = props[idx]
    track_id = rng.integers(0, 1000) if track_id is None else track_id
    tracklet = btrack.btypes.Tracklet(track_id, data)
    tracklet.parent = track_id
    tracklet.root = track_id

    # convert to dictionary {key: [p0,...,pn]}
    properties = {k: [p[k] for p in props] for k in props[0]} if props else {}

    return tracklet, data, properties, track_id


def create_realistic_tracklet(  # noqa: PLR0913
    start_x: float,
    start_y: float,
    dx: float,
    dy: float,
    track_len: float,
    track_ID: int,
) -> btrack.btypes.Tracklet:
    """Create a realistic moving track."""

    data = {
        "x": np.array([start_x + dx * t for t in range(track_len)]),
        "y": np.array([start_y + dy * t for t in range(track_len)]),
        "t": np.arange(track_len),
        "ID": np.array([(track_ID - 1) * track_len + t for t in range(track_len)]),
    }

    objects = btrack.io.objects_from_dict(data)
    return btrack.btypes.Tracklet(track_ID, objects)


def create_test_image(
    boxsize: int = 150,
    ndim: int = 2,
    nobj: int = 10,
    binsize: int = 5,
    *,
    binary: bool = True,
) -> tuple[npt.NDArray, npt.NDArray | None]:
    """Make a test image that ensures that no two pixels are in contact."""

    rng = np.random.default_rng(seed=RANDOM_SEED)

    shape = (boxsize,) * ndim
    img = np.zeros(shape, dtype=np.uint16)

    # return an empty image if we have no objects
    if nobj == 0:
        return img, None

    # split this into voxels
    bins = boxsize // binsize

    def _sample() -> tuple[npt.NDArray, tuple[int]]:
        _img = np.zeros((binsize,) * ndim, dtype=np.uint16)
        _coord = tuple(rng.integers(1, binsize - 1, size=(ndim,)).tolist())
        _img[_coord] = 1
        assert (
            np.sum(_img) == 1
        ), "Test image voxel contains incorrect number of objects."
        return _img, _coord

    # now we update nobj grid positions with a sample
    grid = np.stack(np.meshgrid(*[np.arange(bins)] * ndim), -1).reshape(-1, ndim)

    rbins = rng.choice(grid, size=(nobj,), replace=False)

    # iterate over the bins and add a smaple
    centroids = []
    for v, img_bin in enumerate(rbins):
        sample, point = _sample()
        slices = tuple(slice(b * binsize, b * binsize + binsize, 1) for b in img_bin)
        val = 1 if binary else v + 1
        img[slices] = sample * val

        # shift the actual coordinates back to image space
        point = point + img_bin * binsize
        centroids.append(point)

    # sort the centroids by axis
    centroids_sorted = np.array(centroids)
    centroids_sorted = centroids_sorted[
        np.lexsort([centroids_sorted[:, dim] for dim in range(ndim)][::-1])
    ]

    assert (
        centroids_sorted.shape[0] == nobj
    ), "Number of created centroids != requested in test image."

    vals = np.unique(img)
    assert np.max(vals) == 1 if binary else nobj, "Test image labels are incorrect."
    return img, centroids_sorted


def create_test_segmentation_and_tracks(
    boxsize: int = 128,
    padding: int = 16,
    nframes: int = 10,
    ndim: int = 2,
    *,
    binary: bool = False,
) -> tuple[npt.NDArray, npt.NDArray, list[btrack.btypes.Tracklet]]:
    """Create a test segmentation with four tracks."""

    if ndim not in (btrack.constants.Dimensionality.TWO,):
        raise ValueError("Only 2D-segmentation currently supported.")

    # make a segmentation volume (10, 128, 128) for ndim == 2
    volume = tuple([nframes] + [boxsize] * ndim)
    segmentation = np.zeros(volume, dtype=np.int32)
    ground_truth = np.zeros_like(segmentation)

    dxy = (boxsize - 2 * padding) / nframes

    # create tracks moving from each corner of the segmentation at a constant
    # velocity towards another corner. these tracks do not intersect.
    track_A = create_realistic_tracklet(padding, padding, dxy, 0, nframes, 1)
    track_B = create_realistic_tracklet(
        boxsize - padding, boxsize - padding, -dxy, 0, nframes, 2
    )
    track_C = create_realistic_tracklet(padding, boxsize - padding, 0, -dxy, nframes, 3)
    track_D = create_realistic_tracklet(boxsize - padding, padding, 0, dxy, nframes, 4)

    tracks = [track_A, track_B, track_C, track_D]

    # set the segmentation values
    for track in tracks:
        t, y, x = np.split(track.to_array(properties=["t", "y", "x"]).astype(int), 3, 1)
        segmentation[t, y, x] = 1
        ground_truth[t, y, x] = track.ID

    if not binary:
        for n in range(nframes):
            segmentation[n, ...] = label(segmentation[n, ...])

    return segmentation, ground_truth, tracks


def full_tracker_example(
    objects: list[btrack.btypes.PyTrackObject],
) -> btrack.BayesianTracker:
    # run the tracking
    tracker = btrack.BayesianTracker()
    tracker.configure(CONFIG_FILE)
    tracker.append(objects)
    tracker.volume = ((0, 1600), (0, 1200), (-1e5, 1e5))
    tracker.track(step_size=100)
    tracker.optimize()
    return tracker


def simple_tracker_example() -> tuple[btrack.BayesianTracker, dict[str, Any]]:
    """Run a simple tracker example with some data."""
    x = np.array([200, 201, 202, 203, 204, 207, 208])
    y = np.array([503, 507, 499, 500, 510, 515, 518])
    t = np.array([0, 1, 2, 3, 4, 5, 6])
    z = np.zeros(x.shape)

    objects_dict = {"x": x, "y": y, "z": z, "t": t}
    objects = btrack.io.objects_from_dict(objects_dict)

    tracker = full_tracker_example(objects)
    return tracker, objects_dict
