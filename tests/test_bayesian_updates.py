import math
import numpy as np
import pytest

import _btrack_core
import btrack
from btrack import datasets
from btrack.btypes import PyTrackObject
from btrack.constants import BayesianUpdateFeatures, BayesianUpdates


def test_safe_bayesian_update_boundary_values():
    """Test boundary and standard values in safe_bayesian_update."""
    prob_lost, posterior = _btrack_core.safe_bayesian_update(0.0, 0.9, 0.1)
    assert 0.0 <= posterior <= 1e-15
    assert math.isclose(prob_lost, 1.0, rel_tol=1e-5)

    prob_lost, posterior = _btrack_core.safe_bayesian_update(1.0, 0.9, 0.1)
    assert math.isclose(posterior, 1.0, rel_tol=1e-5)
    assert math.isfinite(prob_lost)

    # Law of Total Probability normalization test
    # prior = 0.5, P(D|A) = 0.9, P(D|~A) = 0.1
    # P(D) = 0.9*0.5 + 0.1*0.5 = 0.5
    # P(A|D) = 0.45 / 0.5 = 0.9
    prob_lost, posterior = _btrack_core.safe_bayesian_update(0.5, 0.9, 0.1)
    assert math.isclose(posterior, 0.9, rel_tol=1e-5)


def test_vmf_likelihood_ratio_properties():
    """Test directional vMF likelihood ratio properties."""
    # Identity
    val = _btrack_core.vmf_likelihood_ratio([1.0, 2.0], [1.0, 2.0], 3.0)
    assert math.isclose(val, 1.0, rel_tol=1e-7)
    # Missing / empty features -> neutral evidence
    assert _btrack_core.vmf_likelihood_ratio([], [], 3.0) == 1.0
    # Orthogonal features -> exp(-kappa)
    ortho = _btrack_core.vmf_likelihood_ratio([1.0, 0.0], [0.0, 1.0], 3.0)
    assert math.isclose(ortho, math.exp(-3.0), rel_tol=1e-5)
    # Opposing features -> exp(-2*kappa)
    opp = _btrack_core.vmf_likelihood_ratio([1.0, 0.0], [-1.0, 0.0], 3.0)
    assert math.isclose(opp, math.exp(-6.0), rel_tol=1e-5)


def test_permutation_invariance_exact_mode():
    """Test that candidate object order within a frame does not affect track output."""
    config_file = datasets.cell_config()

    def run_tracking(cands):
        with btrack.BayesianTracker(verbose=False) as tracker:
            cfg = btrack.config.load_config(config_file)
            cfg.max_search_radius = 50.0
            cfg.update_method = BayesianUpdates.EXACT
            tracker.configure(cfg)

            obj0 = PyTrackObject().from_dict({
                "ID": 0, "t": 0, "x": 100.0, "y": 100.0, "z": 0.0
            })
            tracker.append([obj0] + cands)
            tracker.track()
            return tuple(
                sorted(
                    (len(tr), round(float(tr.x[-1]), 4), round(float(tr.y[-1]), 4))
                    for tr in tracker.tracks
                )
            )

    cands_a = [
        PyTrackObject().from_dict({"ID": 1, "t": 1, "x": 115.0, "y": 100.0, "z": 0.0}),
        PyTrackObject().from_dict({"ID": 2, "t": 1, "x": 92.5, "y": 113.0, "z": 0.0}),
        PyTrackObject().from_dict({"ID": 3, "t": 1, "x": 92.5, "y": 87.0, "z": 0.0}),
    ]
    ref_result = run_tracking(cands_a)

    # Test reverse and shuffled permutations
    cands_b = list(reversed(cands_a))
    assert run_tracking(cands_b) == ref_result

    rng = np.random.default_rng(42)
    for _ in range(5):
        shuffled = list(cands_a)
        rng.shuffle(shuffled)
        assert run_tracking(shuffled) == ref_result


def test_visual_feature_disambiguation():
    """Test that visual appearance features prevent identity swaps during ambiguous proximity."""
    config_file = datasets.cell_config()

    def run_case(enable_visual: bool):
        objs = [
            PyTrackObject().from_dict({
                "ID": 0, "t": 0, "x": 10.0, "y": 10.0, "z": 0.0, "f1": 1.0, "f2": 0.0
            }),
            PyTrackObject().from_dict({
                "ID": 1, "t": 0, "x": 10.0, "y": 20.0, "z": 0.0, "f1": 0.0, "f2": 1.0
            }),
            PyTrackObject().from_dict({
                "ID": 2, "t": 1, "x": 10.0, "y": 12.0, "z": 0.0, "f1": 0.0, "f2": 1.0
            }),
            PyTrackObject().from_dict({
                "ID": 3, "t": 1, "x": 10.0, "y": 18.0, "z": 0.0, "f1": 1.0, "f2": 0.0
            }),
        ]
        with btrack.BayesianTracker(verbose=False) as tracker:
            cfg = btrack.config.load_config(config_file)
            cfg.max_search_radius = 25.0
            cfg.features = ["f1", "f2"]
            cfg.feature_concentration = 5.0
            if enable_visual:
                cfg.tracking_updates = [
                    BayesianUpdateFeatures.MOTION,
                    BayesianUpdateFeatures.VISUAL,
                ]
            else:
                cfg.tracking_updates = [BayesianUpdateFeatures.MOTION]
            tracker.configure(cfg)
            tracker.append(objs)
            tracker.track()
            tracker.optimize()
            return [(tr.ID, list(tr["f1"])) for tr in tracker.tracks]

    motion_only = run_case(enable_visual=False)
    # Motion-only prefers closer spatial proximity (dist 2 vs dist 8) -> swaps identities
    assert any(f1_vals[0] != f1_vals[-1] for _, f1_vals in motion_only)

    visual = run_case(enable_visual=True)
    # Visual updates preserve feature identity
    assert all(f1_vals[0] == f1_vals[-1] for _, f1_vals in visual)
