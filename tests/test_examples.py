from btrack import datasets


def test_pooch_registry():
    """
    Test that `pooch` registry is up to date with remote version.
    This will fail if the remote file does not match the hash hard-coded
    in btrack.datasets.
    """
    registry_file = datasets._remote_registry()
