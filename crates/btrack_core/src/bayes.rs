//! Bayesian belief update functions with numerical stability.

//! Bayesian belief update functions with numerical stability.

/// Safe Bayesian update for binary assignment hypothesis testing.
///
/// Computes the posterior probability of candidate assignment $A$ given evidence $D$,
/// using the Law of Total Probability to evaluate the marginal evidence $P(D)$:
/// $$P(D) = P(D \mid A) P(A) + P(D \mid \neg A) P(\neg A)$$
///
/// Then computes the posterior:
/// $$P(A \mid D) = \frac{P(D \mid A) P(A)}{P(D)}$$
///
/// And the proportional update factor for all remaining mutually exclusive hypotheses $\neg A$:
/// $$\text{update} = \frac{1 - P(A \mid D)}{1 - P(A)}$$
///
/// # Arguments
/// * `prior_assign` - Prior probability of assignment $P(A) \in [0, 1]$.
/// * `prob_assign` - Likelihood of evidence under assignment $P(D \mid A) \in [0, 1]$.
/// * `prob_not_assign` - Likelihood of evidence under non-assignment/clutter $P(D \mid \neg A) \in [0, 1]$.
///
/// # Returns
/// A tuple `(update_multiplier, posterior)` where:
/// * `update_multiplier` is the scaling factor for all other hypotheses in the column.
/// * `posterior` is the updated probability for this candidate.
#[inline]
pub fn safe_bayesian_update(
    prior_assign: f64,
    prob_assign: f64,
    prob_not_assign: f64,
) -> (f64, f64) {
    let p_assign = prob_assign.clamp(0.0, 1.0);
    let p_not_assign = prob_not_assign.clamp(1e-99, 1.0);

    let safe_prior_assign = prior_assign.clamp(1e-99, 1.0 - 1e-15);
    let inv_prior_assign = (1.0 - safe_prior_assign).max(1e-15);

    let pr_dp = p_assign * safe_prior_assign + p_not_assign * inv_prior_assign;
    let posterior = if pr_dp <= 1e-99 {
        safe_prior_assign
    } else {
        (p_assign * safe_prior_assign / pr_dp).clamp(0.0, 1.0)
    };

    let update = ((1.0 - posterior) / inv_prior_assign).max(0.0);

    (update, posterior)
}

/// Simple Bayesian update for binary hypothesis testing in odds form.
///
/// Computes the posterior probability given a prior and likelihood ratio:
/// $$P(A \mid D) = \frac{1}{1 + \left(\frac{1}{\text{prior}} - 1\right) \frac{1 - L}{L}}$$
///
/// # Arguments
/// * `prior` - Prior probability $P(A) \in [0, 1]$.
/// * `likelihood` - Likelihood of evidence given $A$, $L = P(D \mid A) \in [0, 1]$.
#[inline]
pub fn safe_bayesian_update_simple(prior: f64, likelihood: f64) -> f64 {
    let safe_prior = prior.clamp(1e-99, 1.0 - 1e-15);
    let safe_l = likelihood.clamp(1e-99, 1.0);
    let l_neg = (1.0 - safe_l).max(1e-99);
    1.0 / (1.0 + ((1.0 / safe_prior) - 1.0) * (l_neg / safe_l))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_bayesian_update() {
        let (update, posterior) = safe_bayesian_update(0.5, 0.8, 0.01);
        assert!(posterior > 0.5 && posterior <= 1.0);
        assert!(update >= 0.0 && update.is_finite());

        // Verify probability sum invariant: posterior + (1 - prior) * update == 1.0
        let total = posterior + (1.0 - 0.5) * update;
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_zero_prior_safety() {
        let (update, posterior) = safe_bayesian_update(0.0, 0.5, 0.01);
        assert!(!update.is_nan() && !posterior.is_nan());
        assert!(posterior <= 1e-12);
    }

    #[test]
    fn test_one_prior_safety() {
        let (update, posterior) = safe_bayesian_update(1.0, 0.5, 0.01);
        assert!(!update.is_nan() && !posterior.is_nan());
        assert!(!update.is_infinite());
    }

    #[test]
    fn test_likelihood_out_of_bounds_safety() {
        let (update, posterior) = safe_bayesian_update(0.5, 1.5, 0.01);
        assert!(posterior <= 1.0);
        assert!(update >= 0.0);
    }

    #[test]
    fn test_safe_bayesian_update_simple() {
        let post_high = safe_bayesian_update_simple(0.5, 0.9);
        let post_low = safe_bayesian_update_simple(0.5, 0.1);
        assert!(post_high > post_low);
    }
}
