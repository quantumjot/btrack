"""Tests for the Bhattacharyya coefficient PDF function.

These tests validate the pure-Python reference implementation of the formula
used in the C++ bhattacharyya() function in btrack/src/pdf.cc.
"""

import math
import random

import pytest


def bhattacharyya_ref(mu_pred, sigma2, x_obs, accuracy):
    """Pure-Python reference implementation matching pdf.cc::bhattacharyya.

    Parameters
    ----------
    mu_pred : sequence of 3 floats
        Predicted position (Kalman filter mean, first 3 state components).
    sigma2 : sequence of 3 floats
        Diagonal prediction covariance entries covar(i, i) for i in 0..2.
    x_obs : sequence of 3 floats
        Observed position.
    accuracy : float
        Std dev of the isotropic observation Gaussian.

    Returns
    -------
    float
        Bhattacharyya coefficient in (0, 1].
    """
    acc2 = accuracy * accuracy
    mahal = 0.0
    det_term = 0.0
    for i in range(3):
        s2 = sigma2[i]
        d = mu_pred[i] - x_obs[i]
        sum_var = s2 + acc2
        mahal += (d * d) / sum_var
        det_term += math.log(sum_var / (2.0 * math.sqrt(s2) * accuracy))
    D_B = 0.25 * mahal + 0.5 * det_term
    return math.exp(-D_B)


class TestBhattacharyyaReference:
    def test_perfect_overlap(self):
        """d=0 and sigma==accuracy gives BC=1 (identical Gaussians)."""
        acc = 2.0
        bc = bhattacharyya_ref([0, 0, 0], [acc**2] * 3, [0, 0, 0], acc)
        assert bc == pytest.approx(1.0, abs=1e-12)

    def test_zero_displacement_unequal_variance(self):
        """d=0 but sigma != accuracy gives BC < 1."""
        bc = bhattacharyya_ref([0, 0, 0], [1.0] * 3, [0, 0, 0], accuracy=2.0)
        assert 0.0 < bc < 1.0

    def test_monotonic_decrease_with_displacement(self):
        """BC decreases as the prediction moves further from the observation."""
        sigma2 = [1.0] * 3
        acc = 2.0
        obs = [0.0] * 3
        prev_bc = bhattacharyya_ref([0, 0, 0], sigma2, obs, acc)
        for d in [1.0, 2.0, 5.0, 10.0]:
            bc = bhattacharyya_ref([d, 0, 0], sigma2, obs, acc)
            assert bc < prev_bc
            prev_bc = bc

    def test_symmetry(self):
        """Swapping prediction and observation gives the same BC."""
        sigma2 = [1.5, 2.0, 0.5]
        acc = 1.0
        mu = [1.0, -2.0, 3.0]
        x = [0.5, 1.0, 2.5]
        bc_fwd = bhattacharyya_ref(mu, sigma2, x, acc)
        bc_rev = bhattacharyya_ref(x, sigma2, mu, acc)
        assert bc_fwd == pytest.approx(bc_rev, rel=1e-12)

    def test_output_in_range(self):
        """BC is always in (0, 1] for a variety of inputs."""
        rng = random.Random(42)  # noqa: S311
        for _ in range(200):
            mu = [rng.uniform(-10, 10) for _ in range(3)]
            sigma2 = [rng.uniform(0.01, 9.0) for _ in range(3)]
            x = [rng.uniform(-10, 10) for _ in range(3)]
            acc = rng.uniform(0.1, 9.9)
            bc = bhattacharyya_ref(mu, sigma2, x, acc)
            assert 0.0 < bc <= 1.0 + 1e-12

    def test_no_nan_or_inf(self):
        """BC is finite for edge-case inputs."""
        edge_cases = [
            ([0] * 3, [1e-6] * 3, [0] * 3, 0.1),
            ([0] * 3, [100.0] * 3, [0] * 3, 9.9),
            ([50, 50, 50], [1.0] * 3, [0, 0, 0], 2.0),
        ]
        for mu, s2, x, acc in edge_cases:
            bc = bhattacharyya_ref(mu, s2, x, acc)
            assert math.isfinite(bc)
