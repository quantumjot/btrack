"""Validation script to demonstrate the Bayesian update formula error.

This script compares the current (incorrect) implementation with the
mathematically correct version of the Bayesian update formula.
"""
# ruff: noqa: T201, PLR2004, RUF059

import numpy as np

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def current_bayesian_update(
    prior_assign: float, prob_assign: float, prob_not_assign: float
) -> tuple[float, float]:
    """Current (incorrect) implementation from bayes.cc.

    Uses (1. - prob_assign) instead of (1. - safe_prior_assign).
    """
    safe_prior_assign = max(prior_assign, 1e-99)
    inv_prior_assign = 1.0 - safe_prior_assign

    # INCORRECT: Uses (1. - prob_assign) instead of inv_prior_assign
    PrDP = prob_assign * safe_prior_assign + prob_not_assign * (1.0 - prob_assign)
    posterior = prob_assign * (safe_prior_assign / PrDP)
    update = 1.0 + (safe_prior_assign - posterior) / inv_prior_assign

    return update, posterior


def correct_bayesian_update(
    prior_assign: float, prob_assign: float, prob_not_assign: float
) -> tuple[float, float]:
    """
    Correct implementation using proper Bayes' rule.

    P(H|D) = P(D|H) * P(H) / P(D)
    where P(D) = P(D|H) * P(H) + P(D|¬H) * P(¬H)
    """
    safe_prior_assign = max(prior_assign, 1e-10)  # More conservative threshold
    inv_prior_assign = 1.0 - safe_prior_assign

    # CORRECT: Uses inv_prior_assign (i.e., 1. - safe_prior_assign)
    PrDP = prob_assign * safe_prior_assign + prob_not_assign * inv_prior_assign

    # Avoid division by zero
    PrDP = max(PrDP, 1e-15)

    posterior = (prob_assign * safe_prior_assign) / PrDP
    update = (
        1.0 + (safe_prior_assign - posterior) / inv_prior_assign
        if inv_prior_assign > 1e-15
        else 1.0
    )

    return update, posterior


def manual_bayes_rule(
    prior_assign: float, prob_assign: float, prob_not_assign: float
) -> float:
    """
    Manual calculation using textbook Bayes' rule for verification.

    P(H|D) = P(D|H) * P(H) / [P(D|H) * P(H) + P(D|¬H) * P(¬H)]
    """
    prior_not_assign = 1.0 - prior_assign

    # Marginal probability
    prob_data = prob_assign * prior_assign + prob_not_assign * prior_not_assign

    # Posterior using Bayes' rule
    if prob_data < 1e-15:
        return 0.0

    posterior = (prob_assign * prior_assign) / prob_data

    return posterior


def test_specific_cases():
    """Test specific cases to demonstrate the error."""
    print("=" * 80)
    print("SPECIFIC TEST CASES")
    print("=" * 80)

    test_cases = [
        # (prior, likelihood_assign, likelihood_not_assign, description)
        (0.5, 0.8, 0.01, "Balanced prior, high likelihood"),
        (0.1, 0.8, 0.01, "Low prior, high likelihood"),
        (0.9, 0.8, 0.01, "High prior, high likelihood"),
        (0.5, 0.2, 0.01, "Balanced prior, low likelihood"),
        (0.1, 0.2, 0.01, "Low prior, low likelihood"),
        (0.5, 0.5, 0.5, "Balanced prior, equal likelihoods"),
    ]

    for prior, likelihood_assign, likelihood_not_assign, desc in test_cases:
        print(f"\n{desc}")
        print(f"Prior P(H) = {prior:.4f}")
        print(f"Likelihood P(D|H) = {likelihood_assign:.4f}")
        print(f"Likelihood P(D|¬H) = {likelihood_not_assign:.4f}")
        print("-" * 80)

        # Current implementation
        update_curr, posterior_curr = current_bayesian_update(
            prior, likelihood_assign, likelihood_not_assign
        )

        # Correct implementation
        update_corr, posterior_corr = correct_bayesian_update(
            prior, likelihood_assign, likelihood_not_assign
        )

        # Manual verification
        posterior_manual = manual_bayes_rule(
            prior, likelihood_assign, likelihood_not_assign
        )

        print(f"Current posterior:  {posterior_curr:.6f}")
        print(f"Correct posterior:  {posterior_corr:.6f}")
        print(f"Manual posterior:   {posterior_manual:.6f}")
        print(f"Error (current):    {abs(posterior_curr - posterior_manual):.6e}")
        print(f"Error (correct):    {abs(posterior_corr - posterior_manual):.6e}")

        if abs(posterior_curr - posterior_manual) > 0.001:
            print(f"⚠️  SIGNIFICANT ERROR: {abs(posterior_curr - posterior_manual):.4f}")


def test_parameter_sweep():
    """Test across a range of parameters to visualize the error."""
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP ANALYSIS")
    print("=" * 80)

    # Test across range of priors and likelihoods
    priors = np.linspace(0.01, 0.99, 50)
    likelihoods = np.linspace(0.01, 0.99, 50)
    prob_not_assign = 0.01  # Fixed

    errors = np.zeros((len(priors), len(likelihoods)))

    for i, prior in enumerate(priors):
        for j, likelihood in enumerate(likelihoods):
            _, posterior_curr = current_bayesian_update(
                prior, likelihood, prob_not_assign
            )
            posterior_manual = manual_bayes_rule(prior, likelihood, prob_not_assign)
            errors[i, j] = abs(posterior_curr - posterior_manual)

    # Find worst case
    max_error_idx = np.unravel_index(np.argmax(errors), errors.shape)
    max_error = errors[max_error_idx]
    worst_prior = priors[max_error_idx[0]]
    worst_likelihood = likelihoods[max_error_idx[1]]

    print(f"\nMaximum error: {max_error:.6f}")
    print(f"Occurs at: prior={worst_prior:.4f}, likelihood={worst_likelihood:.4f}")
    print(f"Mean absolute error: {np.mean(errors):.6e}")
    print(f"Median absolute error: {np.median(errors):.6e}")
    pct_errors = 100 * np.sum(errors > 0.01) / errors.size
    print(f"Percentage of cases with error > 0.01: {pct_errors:.2f}%")

    # Create visualization
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Error heatmap
        im1 = axes[0].imshow(
            errors,
            aspect="auto",
            origin="lower",
            extent=[likelihoods[0], likelihoods[-1], priors[0], priors[-1]],
            cmap="hot",
        )
        axes[0].set_xlabel("Likelihood P(D|H)")
        axes[0].set_ylabel("Prior P(H)")
        axes[0].set_title("Absolute Error in Posterior Probability")
        plt.colorbar(im1, ax=axes[0], label="|Error|")

        # Cross-section at prior=0.5
        mid_idx = len(priors) // 2
        axes[1].plot(
            likelihoods, errors[mid_idx, :], "b-", linewidth=2, label="Prior = 0.5"
        )

        # Cross-section at prior=0.1
        low_idx = np.argmin(np.abs(priors - 0.1))
        axes[1].plot(
            likelihoods, errors[low_idx, :], "r--", linewidth=2, label="Prior = 0.1"
        )

        # Cross-section at prior=0.9
        high_idx = np.argmin(np.abs(priors - 0.9))
        axes[1].plot(
            likelihoods, errors[high_idx, :], "g:", linewidth=2, label="Prior = 0.9"
        )

        axes[1].set_xlabel("Likelihood P(D|H)")
        axes[1].set_ylabel("Absolute Error")
        axes[1].set_title("Error vs Likelihood for Different Priors")
        axes[1].legend()
        axes[1].grid(visible=True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("bayesian_update_error_analysis.png", dpi=150)
        print("\nVisualization saved to 'bayesian_update_error_analysis.png'")
    else:
        print("\nSkipping visualization (matplotlib not available)")


def test_sequential_updates():
    """Test how errors accumulate over sequential Bayesian updates."""
    print("\n" + "=" * 80)
    print("SEQUENTIAL UPDATE ANALYSIS")
    print("=" * 80)

    n_updates = 10
    prob_not_assign = 0.01

    # Start with uniform prior
    prior_current = 0.5
    prior_correct = 0.5

    # Sequence of likelihoods (simulating tracking over time)
    likelihoods = [0.7, 0.8, 0.6, 0.9, 0.5, 0.8, 0.7, 0.6, 0.8, 0.7]

    print(f"\nInitial prior: {prior_current:.4f}")
    print(f"Sequence of likelihoods: {likelihoods}")
    print("\nIteration | Current Impl | Correct Impl | Error")
    print("-" * 60)

    errors = []

    for i, likelihood in enumerate(likelihoods):
        # Update with current implementation
        _, posterior_current = current_bayesian_update(
            prior_current, likelihood, prob_not_assign
        )

        # Update with correct implementation
        _, posterior_correct = correct_bayesian_update(
            prior_correct, likelihood, prob_not_assign
        )

        error = abs(posterior_current - posterior_correct)
        errors.append(error)

        print(
            f"{i + 1:9d} | {posterior_current:12.6f} | "
            f"{posterior_correct:12.6f} | {error:.6e}"
        )

        # Use posterior as next prior
        prior_current = posterior_current
        prior_correct = posterior_correct

    print(f"\nFinal error after {n_updates} updates: {errors[-1]:.6e}")
    print(f"Maximum error during sequence: {max(errors):.6e}")
    print(f"Mean error during sequence: {np.mean(errors):.6e}")


def main():
    """Run all validation tests."""
    print("\n" + "=" * 80)
    print("BAYESIAN UPDATE VALIDATION SCRIPT")
    print("Demonstrating the error in btrack/src/bayes.cc:26")
    print("=" * 80)

    test_specific_cases()
    test_sequential_updates()
    test_parameter_sweep()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("The current implementation uses (1. - prob_assign) instead of")
    print("(1. - safe_prior_assign) in the marginal probability calculation.")
    print("This causes incorrect posterior probabilities, especially when:")
    print("  1. Prior and likelihood have different magnitudes")
    print("  2. Sequential updates accumulate errors")
    print("  3. Extreme prior values (very high or very low)")
    print("\nThe error can be significant (>0.1 in some cases) and will")
    print("affect tracking accuracy and object association decisions.")
    print("=" * 80)


if __name__ == "__main__":
    main()
