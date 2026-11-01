//! Bayesian belief update functions with numerical stability.

/// Safe Bayesian update matching C++ `BayesianUpdateFunctions::safe_bayesian_update`.
///
/// Returns `(update_multiplier, posterior)`.
#[inline]
pub fn safe_bayesian_update(
    prior_assign: f64,
    prob_assign: f64,
    prob_not_assign: f64,
) -> (f64, f64) {
    let safe_prior_assign = prior_assign.max(1e-99);
    let inv_prior_assign = 1.0 - safe_prior_assign;

    let pr_dp = prob_assign * safe_prior_assign + prob_not_assign * (1.0 - prob_assign);
    let posterior = prob_assign * (safe_prior_assign / pr_dp);
    let update = 1.0 + (safe_prior_assign - posterior) / inv_prior_assign;

    (update, posterior)
}

/// Simple Bayesian update matching C++ `BayesianUpdateFunctions::safe_bayesian_update_simple`.
#[inline]
pub fn safe_bayesian_update_simple(prior: f64, likelihood: f64) -> f64 {
    let safe_prior = prior.max(1e-99);
    1.0 / (1.0 + ((1.0 / safe_prior) - 1.0) * ((1.0 - safe_prior) / likelihood))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_bayesian_update() {
        let (update, posterior) = safe_bayesian_update(0.5, 0.8, 0.01);
        assert!(posterior > 0.5 && posterior <= 1.0);
        assert!(update.is_finite());
    }

    #[test]
    fn test_zero_prior_safety() {
        let (update, posterior) = safe_bayesian_update(0.0, 0.5, 0.01);
        assert!(!update.is_nan() && !posterior.is_nan());
    }
}
