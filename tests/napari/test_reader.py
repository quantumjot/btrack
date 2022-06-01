import pytest

# Only run tests if napari is installed
napari_reader = pytest.importorskip("btrack.napari.reader")


def test_reader(hdf5_file_path_or_paths):
    reader = napari_reader.get_reader(hdf5_file_path_or_paths)
    assert reader is not None

    tracks = reader(hdf5_file_path_or_paths)
    assert isinstance(tracks, list)
    # TODO: update the HDF file so that it has some tracks that can be read.
    # For now this just checks that the reader can read an empty HDF5 file
    # with no tracks in it
    assert len(tracks) == 0
