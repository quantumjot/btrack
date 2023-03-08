import os
from pathlib import Path

import numpy as np
import pytest

import btrack

from ._utils import (
    create_test_object,
    create_test_properties,
    full_tracker_example,
    simple_tracker_example,
)


def test_hdf5_write(hdf5_file_path, test_objects):
    """Test writing an HDF5 file with some objects."""
    # now try to read those objects and compare with those used to write
    with btrack.io.HDF5FileHandler(hdf5_file_path, "r") as h:
        objects_from_file = h.objects

    properties = ["x", "y", "z", "t", "label", "ID"]

    for orig, read in zip(test_objects, objects_from_file):
        for p in properties:
            # use all close, since h5 file stores in float32 default
            np.testing.assert_allclose(getattr(orig, p), getattr(read, p))


def test_hdf5_write_with_properties(hdf5_file_path):
    """Test writing an HDF5 file with some objects with additional properties."""

    objects = []
    for i in range(10):
        obj, _ = create_test_object(id=i)
        obj.properties = create_test_properties()
        objects.append(obj)

    with btrack.io.HDF5FileHandler(hdf5_file_path, "w") as h:
        h.write_objects(objects)

    # now try to read those objects and compare with those used to write
    with btrack.io.HDF5FileHandler(hdf5_file_path, "r") as h:
        objects_from_file = h.objects

    extra_props = list(create_test_properties().keys())

    properties = ["x", "y", "z", "t", "label", "ID"]

    for orig, read in zip(objects, objects_from_file):
        for p in properties:
            # use all close, since h5 file stores in float32 default
            np.testing.assert_allclose(getattr(orig, p), getattr(read, p))
        for p in extra_props:
            np.testing.assert_allclose(orig.properties[p], read.properties[p])


@pytest.mark.parametrize("export_format", ["", ".csv", ".h5"])
def test_tracker_export(tmp_path, export_format):
    """Test that file export works using the `export_delegator`."""

    tracker, _ = simple_tracker_example()

    export_filename = f"test{export_format}"

    # string type path
    fn = os.path.join(tmp_path, export_filename)
    tracker.export(fn, obj_type="obj_type_1")

    # Pathlib type path
    fn = Path(tmp_path) / export_filename
    tracker.export(fn, obj_type="obj_type_1")

    if export_format:
        assert os.path.exists(fn)


@pytest.mark.parametrize("shuffle_objects", [False, True])
def test_write_tracks_only(
    test_real_objects, hdf5_file_path, default_rng, shuffle_objects
):
    """Test writing tracks only using the file handler."""

    if shuffle_objects:
        default_rng.shuffle(test_real_objects)

    # tracker, _ = simple_tracker_example()
    tracker = full_tracker_example(test_real_objects)
    tracks = tracker.tracks

    with btrack.io.HDF5FileHandler(hdf5_file_path, "w") as h:
        h.write_tracks(tracks)

    # now try to read those objects and compare with those used to write
    with btrack.io.HDF5FileHandler(hdf5_file_path, "r") as h:
        tracks_from_file = h.tracks

    for orig, read in zip(tracks, tracks_from_file):
        assert isinstance(orig, btrack.btypes.Tracklet)
        assert isinstance(read, btrack.btypes.Tracklet)

        gt_track = orig.to_dict()
        io_track = read.to_dict()

        for key, gt_value in gt_track.items():
            io_value = io_track[key]
            np.testing.assert_allclose(gt_value, io_value)


def test_write_lbep(tmp_path, test_real_objects):
    """Test writing the LBEP file."""
    tracker = full_tracker_example(test_real_objects)
    tracks = tracker.tracks

    fn = Path(tmp_path) / "LBEP_test.txt"
    btrack.io.export_LBEP(fn, tracker.tracks)

    # check that the file contains the correct number of lines
    with open(fn, "r") as lbep_file:
        entries = lbep_file.readlines()
    assert len(entries) == len(tracks)
    # and that the LBEP entries match
    for entry in entries:
        lbep = [int(e) for e in entry.strip("/n").split()]
        track = next(filter(lambda t: t.ID == lbep[0], tracks))
        assert lbep == [track.ID, track.start, track.stop, track.parent]


def test_write_hdf_segmentation(hdf5_file_path):
    """Test writing a segmentation to the hdf file."""
    segmentation = np.random.randint(0, 255, size=(100, 64, 64))
    with btrack.io.HDF5FileHandler(hdf5_file_path, "w") as h:
        h.write_segmentation(segmentation)

    with btrack.io.HDF5FileHandler(hdf5_file_path, "r") as h:
        segmentation_from_file = h.segmentation
    np.testing.assert_equal(segmentation, segmentation_from_file)
