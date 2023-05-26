import platform
from pathlib import Path

import pytest

import btrack
from btrack.constants import BTRACK_LIB_PATH
from btrack.libwrapper import load_library


def test_load_library():
    """Test loading the shared library."""
    load_library(BTRACK_LIB_PATH)


def test_fails_load_library_debug(tmp_path):
    """Test loading a fake shared library."""
    fake_lib_filename = Path(tmp_path) / "fakelib"
    with pytest.raises(FileNotFoundError):
        load_library(fake_lib_filename)


def test_debug_info():
    """Test debugging info."""
    system_info = btrack.SystemInformation()

    assert isinstance(repr(system_info), str)
    assert system_info.btrack_version == btrack.__version__
    assert system_info.system_platform == platform.platform()
    assert system_info.system_python == platform.python_version()
