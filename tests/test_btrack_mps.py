import os
import numpy as np
import pandas as pd
import pytest

import btrack
from btrack import constants
from btrack.io.utils import objects_from_array
from btrack import datasets

# --- Define path to local CSV data ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DATA_FILE = os.path.join(BASE_DIR, "_test_data", "test_data.csv")


@pytest.fixture
def test_objects():
    """Load test objects from CSV file."""
    if not os.path.exists(CSV_DATA_FILE):
        pytest.skip(f"CSV data file not found at {CSV_DATA_FILE}")
    
    df = pd.read_csv(CSV_DATA_FILE)
    columns_for_numpy = ['t', 'x', 'y', 'z', 'label']
    
    if not all(col in df.columns for col in columns_for_numpy):
        pytest.skip(f"CSV file must contain columns: {', '.join(columns_for_numpy)}")
    
    objects_nparray = df[columns_for_numpy].to_numpy()
    objects = objects_from_array(objects_nparray)
    
    if not objects:
        pytest.skip("No objects created from CSV data")
    
    return objects, df


@pytest.fixture
def tracker_config():
    """Get tracker configuration."""
    return datasets.cell_config()


def _run_tracking(objects, df, config_file, update_method):
    """Helper function to run tracking with specified update method."""
    with btrack.BayesianTracker() as tracker:
        tracker.configure(config_file)
        tracker.update_method = update_method
        tracker.max_search_radius = 50
        tracker.tracking_updates = ["MOTION"]
        
        min_x, max_x = df['x'].min() - 10, df['x'].max() + 10
        min_y, max_y = df['y'].min() - 10, df['y'].max() + 10
        min_z, max_z = df['z'].min() - 5, df['z'].max() + 5
        tracker.volume = ((min_x, max_x), (min_y, max_y), (min_z, max_z))
        
        tracker.append(objects)
        tracker.track()
        tracker.optimize()
        
        return tracker.tracks


def test_mps_method_available():
    """Test that MPS update method is available."""
    assert hasattr(constants.BayesianUpdates, 'MPS')
    assert constants.BayesianUpdates.MPS in list(constants.BayesianUpdates)


def test_mps_vs_exact_tracking(test_objects, tracker_config):
    """Test that MPS method produces similar results to EXACT method."""
    objects, df = test_objects
    
    # Run tracking with EXACT method
    tracks_exact = _run_tracking(objects, df, tracker_config, constants.BayesianUpdates.EXACT)
    
    # Run tracking with MPS method
    tracks_mps = _run_tracking(objects, df, tracker_config, constants.BayesianUpdates.MPS)
    
    # Basic assertions
    assert len(tracks_exact) > 0, "EXACT method should produce tracks"
    assert len(tracks_mps) > 0, "MPS method should produce tracks"
    assert len(tracks_exact) == len(tracks_mps), "Both methods should produce same number of tracks"
    
    # Compare track properties
    for track_exact, track_mps in zip(tracks_exact, tracks_mps):
        assert len(track_exact) == len(track_mps), f"Track lengths should match"
        
        # Compare positions (allow small numerical differences)
        np.testing.assert_allclose(track_exact.x, track_mps.x, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(track_exact.y, track_mps.y, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(track_exact.z, track_mps.z, rtol=1e-10, atol=1e-10)
        
        # Time points should be identical
        np.testing.assert_array_equal(track_exact.t, track_mps.t)


def test_mps_tracker_configuration(test_objects, tracker_config):
    """Test that MPS method can be properly configured."""
    objects, df = test_objects
    
    with btrack.BayesianTracker() as tracker:
        tracker.configure(tracker_config)
        tracker.update_method = constants.BayesianUpdates.MPS
        
        # Verify configuration
        assert tracker.update_method == constants.BayesianUpdates.MPS
        assert tracker.configuration.update_method == constants.BayesianUpdates.MPS


def test_mps_tracking_completion(test_objects, tracker_config):
    """Test that MPS tracking completes successfully."""
    objects, df = test_objects
    
    tracks = _run_tracking(objects, df, tracker_config, constants.BayesianUpdates.MPS)
    
    assert len(tracks) > 0, "MPS tracking should produce tracks"
    
    # Verify track properties
    for track in tracks:
        assert len(track) > 0, "Each track should have at least one point"
        assert hasattr(track, 'ID'), "Track should have ID attribute"
        assert hasattr(track, 'x'), "Track should have x coordinates"
        assert hasattr(track, 'y'), "Track should have y coordinates"
        assert hasattr(track, 'z'), "Track should have z coordinates"
        assert hasattr(track, 't'), "Track should have time points"
