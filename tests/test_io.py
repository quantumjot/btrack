import os

import numpy as np
from _utils import create_test_object

import btrack


def test_hdf5_write(tmp_path):
    """Test writing an HDF5 file with some objects."""
    fn = os.path.join(tmp_path, "test.h5")

    objects = []
    for i in range(10):
        obj, _ = create_test_object(id=i)
        objects.append(obj)

    with btrack.dataio.HDF5FileHandler(fn, 'w') as h:
        h.write_objects(objects)

    # now try to read those objects and compare with those used to write
    with btrack.dataio.HDF5FileHandler(fn, 'r') as h:
        objects_from_file = h.objects

    properties = ['x', 'y', 'z', 't', 'label', 'ID']

    for orig, read in zip(objects, objects_from_file):
        for p in properties:
            # use all close, since h5 file stores in float32 default
            np.testing.assert_allclose(getattr(orig, p), getattr(read, p))
