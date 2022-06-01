from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import btrack

CONFIG_FILE = (
    Path(__file__).resolve().parent.parent / "models" / "cell_config.json"
)


def create_test_object(
    id: Optional[int] = None,
) -> Tuple[btrack.btypes.PyTrackObject, Dict[str, Any]]:
    """Create a test object."""
    data = {
        "ID": np.random.randint(0, 1000) if id is None else int(id),
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


def create_test_properties() -> Dict[str, float]:
    properties = {
        "speed": np.random.uniform(0.0, 1.0),
        "circularity": np.random.uniform(0.0, 1.0),
        "reporter": np.random.uniform(0.0, 1.0),
    }
    return properties


def create_test_tracklet(
    track_len: int,
) -> Tuple[
    btrack.btypes.Tracklet,
    List[btrack.btypes.PyTrackObject],
    List[Dict[str, Any]],
    int,
]:
    """Create a test track."""
    data = [create_test_object()[0] for i in range(track_len)]
    props = [create_test_properties() for i in range(track_len)]
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


def create_realistc_tracklet(
    start_x: float,
    start_y: float,
    dx: float,
    dy: float,
    track_len: float,
    track_ID: int,
) -> btrack.btypes.Tracklet:
    """Create a realistic moving track."""
    data = {
        "x": start_x,
        "y": start_y,
        "z": 0,
        "t": 0,
        "ID": (track_ID - 1) * track_len,
    }

    objects = [
        btrack.btypes.PyTrackObject.from_dict(data),
    ]

    for t in range(1, track_len):
        new_data = {
            "x": data["x"] + dx,
            "y": data["y"] + dy,
            "t": t,
            "ID": (track_ID - 1) * track_len + t,
        }
        data.update(new_data)
        objects.append(
            btrack.btypes.PyTrackObject.from_dict(data),
        )

    return btrack.btypes.Tracklet(track_ID, objects)


def create_test_image(
    boxsize: int = 150,
    ndim: int = 2,
    nobj: int = 10,
    binsize: int = 5,
    binary: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
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


def create_test_segmentation_and_tracks(
    nframes: int = 10,
    ndim: int = 2,
    binary: bool = False,
) -> Tuple[np.ndarray, List[btrack.btypes.Tracklet]]:
    """Create a test segmentation with four tracks."""

    # make a segmentation volume (10, 128, 128) for ndim == 2
    volume = tuple([nframes] + [128] * ndim)
    segmentation = np.zeros(volume, dtype=np.int32)

    dxy = (volume[1] - 32.0) / nframes

    # create tracks moving from each corner of the segmentation at a constant
    # velocity towards another corner
    track_A = create_realistc_tracklet(16, 16, dxy, 0, nframes, 1)
    track_B = create_realistc_tracklet(128 - 16, 128 - 16, -dxy, 0, nframes, 2)
    track_C = create_realistc_tracklet(16, 128 - 16, 0, -dxy, nframes, 3)
    track_D = create_realistc_tracklet(128 - 16, 16, 0, dxy, nframes, 4)

    tracks = [track_A, track_B, track_C, track_D]

    # set the segmentation values
    for track in tracks:
        coords = [track.t, track.y, track.x]
        segmentation[coords] = 1

    return segmentation, tracks


def full_tracker_example(
    objects: List[btrack.btypes.PyTrackObject],
) -> btrack.BayesianTracker:
    # run the tracking
    tracker = btrack.BayesianTracker()
    tracker.configure(CONFIG_FILE)
    tracker.append(objects)
    tracker.volume = ((0, 1600), (0, 1200), (-1e5, 1e5))
    tracker.track_interactive(step_size=100)
    tracker.optimize()
    return tracker


def simple_tracker_example() -> Tuple[btrack.BayesianTracker, Dict[str, Any]]:
    """Run a simple tracker example with some data."""
    x = np.array([200, 201, 202, 203, 204, 207, 208])
    y = np.array([503, 507, 499, 500, 510, 515, 518])
    t = np.array([0, 1, 2, 3, 4, 5, 6])
    z = np.zeros(x.shape)

    objects_dict = {"x": x, "y": y, "z": z, "t": t}
    objects = btrack.dataio.objects_from_dict(objects_dict)

    tracker = full_tracker_example(objects)
    return tracker, objects_dict
