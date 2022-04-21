import pytest

# Only run tests is napari is installed
napari_reader = pytest.importorskip("btrack.napari.reader")


def test_get_reader_pass():
    reader = napari_reader.get_reader("fake.file")
    assert reader is not None
