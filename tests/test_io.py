import btrack

from ._utils import (
    create_test_object,
    create_test_properties,
    full_tracker_example,
    simple_tracker_example,
)

import os
from pathlib import Path

import numpy as np
import pytest


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
        obj, _ = create_test_object(test_id=i)
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


@pytest.mark.parametrize("frac_dummies", [0.1, 0.5, 0.9])
def test_hdf5_write_dummies(hdf5_file_path, test_objects, frac_dummies):
    """Test writing tracks with a variable proportion of dummy objects."""

    num_dummies = int(len(test_objects) * frac_dummies)

    for obj in test_objects[:num_dummies]:
        obj.dummy = True
        obj.ID = -(obj.ID + 1)

    track_id = 1
    track_with_dummies = btrack.btypes.Tracklet(track_id, test_objects)
    track_with_dummies.root = track_id
    track_with_dummies.parent = track_id

    # write them out
    with btrack.io.HDF5FileHandler(hdf5_file_path, "w") as h:
        h.write_tracks(
            [
                track_with_dummies,
            ]
        )

    # read them in
    with btrack.io.HDF5FileHandler(hdf5_file_path, "r") as h:
        tracks_from_file = h.tracks
    objects_from_file = tracks_from_file[0]._data

    assert sum(obj.dummy for obj in objects_from_file) == num_dummies


@pytest.mark.parametrize("export_format", ["", ".csv", ".h5", ".geff"])
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
        track = next(filter(lambda t: lbep[0] == t.ID, tracks))
        assert lbep == [track.ID, track.start, track.stop, track.parent]


def test_write_hdf_segmentation(hdf5_file_path):
    """Test writing a segmentation to the hdf file."""
    segmentation = np.random.randint(0, 255, size=(100, 64, 64))
    with btrack.io.HDF5FileHandler(hdf5_file_path, "w") as h:
        h.write_segmentation(segmentation)

    with btrack.io.HDF5FileHandler(hdf5_file_path, "r") as h:
        segmentation_from_file = h.segmentation
    np.testing.assert_equal(segmentation, segmentation_from_file)


def test_hdf_tree(hdf5_file_path, caplog):
    """Test that the tree function iterates over the files and writes the output
    to the logger."""
    n_log_entries = len(caplog.records)

    # first test with an empty tree
    btrack.io.hdf._h5_tree({})

    assert len(caplog.records) == n_log_entries

    with btrack.io.HDF5FileHandler(hdf5_file_path, "r") as hdf:
        hdf.tree()

    n_expected_entries = 8
    assert len(caplog.records) == n_log_entries + n_expected_entries


def test_geff_roundtrip(tmp_path, test_real_objects):
    """Test GEFF export and import roundtrip."""
    pytest.importorskip("geff")

    # Create tracker with test data
    tracker = full_tracker_example(test_real_objects)
    original_tracks = tracker.tracks

    # Export to GEFF
    geff_file = Path(tmp_path) / "test_tracks.geff"
    btrack.io.export_GEFF(geff_file, original_tracks)

    assert geff_file.exists()

    # Import from GEFF
    imported_objects = btrack.io.import_GEFF(geff_file)

    # Check that we got objects back
    assert len(imported_objects) > 0

    # Verify object properties
    original_objects = []
    for track in original_tracks:
        original_objects.extend(track._data)

    # Sort both lists by ID and time for comparison
    original_objects.sort(key=lambda obj: (obj.ID, obj.t))
    imported_objects.sort(key=lambda obj: (obj.ID, obj.t))

    assert len(imported_objects) == len(original_objects)

    # Check key properties match
    for orig, imported in zip(original_objects, imported_objects):
        np.testing.assert_allclose(orig.x, imported.x, rtol=1e-5)
        np.testing.assert_allclose(orig.y, imported.y, rtol=1e-5)
        np.testing.assert_allclose(orig.z, imported.z, rtol=1e-5)
        assert orig.t == imported.t
        assert orig.label == imported.label
        assert orig.dummy == imported.dummy


def test_geff_export_simple(tmp_path):
    """Test GEFF export with simple track data."""
    pytest.importorskip("geff")

    # Create simple test objects
    objects = []
    for i in range(5):
        obj, _ = create_test_object(test_id=i)
        obj.t = i  # Set sequential time points
        objects.append(obj)

    # Create a simple tracklet
    track = btrack.btypes.Tracklet(1, objects)
    tracks = [track]

    # Export to GEFF
    geff_file = Path(tmp_path) / "simple_track.geff"
    btrack.io.export_GEFF(geff_file, tracks)

    assert geff_file.exists()


def test_geff_import_nonexistent_file():
    """Test GEFF import with non-existent file."""
    with pytest.raises(FileNotFoundError):
        btrack.io.import_GEFF("nonexistent.geff")


def test_geff_export_empty_tracks(tmp_path):
    """Test GEFF export with empty tracks list."""
    geff_file = Path(tmp_path) / "empty.geff"
    btrack.io.export_GEFF(geff_file, [])

    # Should not create file or should create empty file
    # The exact behavior depends on GEFF library implementation


def test_geff_export_with_lineage(tmp_path):
    """Test GEFF export with parent-child relationships."""
    pytest.importorskip("geff")

    # Create parent track
    parent_objects = []
    for i in range(3):
        obj, _ = create_test_object(test_id=i)
        obj.t = i
        parent_objects.append(obj)
    parent_track = btrack.btypes.Tracklet(1, parent_objects)

    # Create child track
    child_objects = []
    for i in range(2):
        obj, _ = create_test_object(test_id=i + 10)
        obj.t = i + 3  # Start after parent
        child_objects.append(obj)
    child_track = btrack.btypes.Tracklet(2, child_objects)
    child_track.parent = 1  # Set parent relationship

    tracks = [parent_track, child_track]

    # Export to GEFF
    geff_file = Path(tmp_path) / "lineage_tracks.geff"
    btrack.io.export_GEFF(geff_file, tracks)

    assert geff_file.exists()


def test_geff_delegator_integration(tmp_path):
    """Test GEFF export through the export_delegator."""
    pytest.importorskip("geff")

    tracker, _ = simple_tracker_example()

    # Test .geff extension
    geff_file = Path(tmp_path) / "delegator_test.geff"
    btrack.io.export_delegator(geff_file, tracker, obj_type="obj_type_1")
    assert geff_file.exists()

    # Test .zarr extension
    zarr_file = Path(tmp_path) / "delegator_test.zarr"
    btrack.io.export_delegator(zarr_file, tracker, obj_type="obj_type_1")
    assert zarr_file.exists()


def test_geff_tracks_roundtrip(tmp_path, test_real_objects):
    """Test GEFF track export and import roundtrip, verifying track structure."""
    pytest.importorskip("geff")

    # Create tracker with test data
    tracker = full_tracker_example(test_real_objects)
    original_tracks = tracker.tracks

    # Export to GEFF
    geff_file = Path(tmp_path) / "test_track_structure.geff"
    btrack.io.export_GEFF(geff_file, original_tracks)

    assert geff_file.exists()

    # Import tracks (not just objects)
    imported_tracks = btrack.io.import_GEFF_tracks(geff_file)

    # Check that we got tracks back
    assert len(imported_tracks) > 0
    assert len(imported_tracks) == len(original_tracks)

    # Sort both lists by track ID for comparison
    original_tracks.sort(key=lambda t: t.ID)
    imported_tracks.sort(key=lambda t: t.ID)

    # Verify track structure is preserved
    for orig_track, imported_track in zip(original_tracks, imported_tracks):
        # Check track properties
        assert orig_track.ID == imported_track.ID
        assert len(orig_track._data) == len(imported_track._data)

        # Check parent relationships
        assert orig_track.parent == imported_track.parent

        # Check that objects within tracks are properly ordered by time
        imported_times = [obj.t for obj in imported_track._data]
        assert imported_times == sorted(imported_times), (
            "Objects should be sorted by time"
        )

        # Verify object properties within each track
        for orig_obj, imported_obj in zip(orig_track._data, imported_track._data):
            np.testing.assert_allclose(orig_obj.x, imported_obj.x, rtol=1e-5)
            np.testing.assert_allclose(orig_obj.y, imported_obj.y, rtol=1e-5)
            np.testing.assert_allclose(orig_obj.z, imported_obj.z, rtol=1e-5)
            assert orig_obj.t == imported_obj.t
            assert orig_obj.label == imported_obj.label
            assert orig_obj.dummy == imported_obj.dummy


def test_geff_lineage_preservation(tmp_path):
    """Test that parent-child relationships are preserved in GEFF roundtrip."""
    pytest.importorskip("geff")

    # Create parent track
    parent_objects = []
    for i in range(3):
        obj, _ = create_test_object(test_id=i)
        obj.t = i
        parent_objects.append(obj)
    parent_track = btrack.btypes.Tracklet(1, parent_objects)

    # Create child track with parent relationship
    child_objects = []
    for i in range(2):
        obj, _ = create_test_object(test_id=i + 10)
        obj.t = i + 3  # Start after parent
        child_objects.append(obj)
    child_track = btrack.btypes.Tracklet(2, child_objects)
    child_track.parent = 1  # Set parent relationship

    original_tracks = [parent_track, child_track]

    # Export and reimport
    geff_file = Path(tmp_path) / "lineage_test.geff"
    btrack.io.export_GEFF(geff_file, original_tracks)
    imported_tracks = btrack.io.import_GEFF_tracks(geff_file)

    # Constants for test track IDs
    PARENT_TRACK_ID = 1
    CHILD_TRACK_ID = 2

    # Verify parent-child relationship is preserved
    imported_tracks.sort(key=lambda t: t.ID)
    parent_imported, child_imported = imported_tracks

    assert parent_imported.ID == PARENT_TRACK_ID
    assert child_imported.ID == CHILD_TRACK_ID
    assert child_imported.parent == PARENT_TRACK_ID
    assert (
        parent_imported.parent is None or parent_imported.parent == PARENT_TRACK_ID
    )  # Parent of root can be None or itself
