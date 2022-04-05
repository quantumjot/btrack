from btrack import datasets


def test_pooch_registry():
    """Test that `pooch` registry is up to date with remote version."""
    registry_file = datasets._remote_registry()
