import os
from pathlib import Path

import numpy as np
import pytest

import btrack

from ._utils import (
    create_test_object,
    create_test_properties,
    simple_tracker_example,
)


def test_hdf5_write(hdf5_file_path, test_objects):
    """Test writing an HDF5 file with some objects."""
    # now try to read those objects and compare with those used to write
    with btrack.dataio.HDF5FileHandler(hdf5_file_path, "r") as h:
        objects_from_file = h.objects

    properties = ["x", "y", "z", "t", "label", "ID"]

    for orig, read in zip(test_objects, objects_from_file):
        for p in properties:
            # use all close, since h5 file stores in float32 default
            np.testing.assert_allclose(getattr(orig, p), getattr(read, p))


def test_hdf5_write_with_properties(tmp_path):
    """Test writing an HDF5 file with some objects with additional properties."""
    fn = os.path.join(tmp_path, "test.h5")

    objects = []
    for i in range(10):
        obj, _ = create_test_object(id=i)
        obj.properties = create_test_properties()
        objects.append(obj)

    with btrack.dataio.HDF5FileHandler(fn, "w") as h:
        h.write_objects(objects)

    # now try to read those objects and compare with those used to write
    with btrack.dataio.HDF5FileHandler(fn, "r") as h:
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
