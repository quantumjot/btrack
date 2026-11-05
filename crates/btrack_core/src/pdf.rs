//! Probability density and similarity functions.

use nalgebra::Vector3;
use statrs::function::erf::erf;

use crate::types::Prediction;

const ROOT_TWO: f64 = std::f64::consts::SQRT_2;
const ROOT_TWO_PI: f64 = 2.5066282746310002; // sqrt(2 * PI)

/// Multivariate Gaussian probability density function under independence (diagonal covariance).
pub fn multivariate_normal(pred: &Prediction, obj_pos: &Vector3<f64>) -> f64 {
    let mut prob_density = 1.0;
    for axis in 0..3 {
        let var = pred.covar[(axis, axis)];
        let std_dev = var.sqrt();
        let diff = obj_pos[axis] - pred.mu[axis];
        let p = (1.0 / (ROOT_TWO_PI * std_dev)) * (-0.5 * (diff * diff) / var).exp();
        prob_density *= p;
    }
    prob_density
}

/// Integrates the 3D Gaussian over an observation window `[-accuracy, +accuracy]` using erf.
///
/// Matches C++ `ProbabilityDensityFunctions::multivariate_erf`.
pub fn multivariate_erf(pred: &Prediction, obj_pos: &Vector3<f64>, accuracy: f64) -> f64 {
    let mut phi = 1.0;

    for axis in 0..3 {
        let var = pred.covar[(axis, axis)].max(1e-12);
        let std_x = var.sqrt();
        let d_x = obj_pos[axis] - pred.mu[axis];

        let phi_x = erf((d_x + accuracy) / (std_x * ROOT_TWO))
            - erf((d_x - accuracy) / (std_x * ROOT_TWO));

        phi *= 0.5 * phi_x;
    }

    if phi.is_nan() {
        0.0
    } else {
        phi
    }
}

/// Precomputed inverse denominator for fast erf calculation across many objects:
/// `inv_denom[axis] = 1.0 / (std_x * ROOT_TWO)`.
#[inline]
pub fn precompute_erf_inv_denom(pred: &Prediction) -> [f64; 3] {
    let mut inv = [0.0; 3];
    for axis in 0..3 {
        let var = pred.covar[(axis, axis)].max(1e-12);
        let std_x = var.sqrt();
        inv[axis] = 1.0 / (std_x * ROOT_TWO);
    }
    inv
}

/// Fast multivariate erf calculation using precomputed means and inverse denominators.
#[inline]
pub fn multivariate_erf_fast(
    mu: &[f64; 3],
    inv_denom: &[f64; 3],
    obj_pos: &[f64; 3],
    accuracy: f64,
) -> f64 {
    let mut phi = 1.0;

    for axis in 0..3 {
        let d_x = obj_pos[axis] - mu[axis];
        let phi_x = erf((d_x + accuracy) * inv_denom[axis])
            - erf((d_x - accuracy) * inv_denom[axis]);
        phi *= 0.5 * phi_x;
    }

    if phi.is_nan() {
        0.0
    } else {
        phi
    }
}

/// Computes the Euclidean L2 norm of a feature slice.
#[inline]
pub fn feature_norm(f: &[f64]) -> f64 {
    f.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Computes the directional von Mises-Fisher (vMF) feature likelihood ratio with precomputed L2 norms:
/// `exp(-kappa * (1.0 - cos_sim))`.
///
/// * When features are identical (`cos_sim == 1.0`), returns `1.0`.
/// * When features are missing or zero-norm, returns `1.0` (uninformative neutral evidence).
/// * When features diverge, decays exponentially at rate `kappa`.
#[inline]
pub fn vmf_likelihood_ratio_fast(
    f1: &[f64],
    norm1: f64,
    f2: &[f64],
    norm2: f64,
    kappa: f64,
) -> f64 {
    if f1.is_empty() || f2.is_empty() || f1.len() != f2.len() {
        return 1.0;
    }

    let mag = norm1 * norm2;
    if mag <= 1e-12 {
        return 1.0;
    }

    let dot: f64 = f1.iter().zip(f2.iter()).map(|(&a, &b)| a * b).sum();
    let cos_sim = (dot / mag).clamp(-1.0, 1.0);
    (-kappa * (1.0 - cos_sim)).exp()
}

/// Computes the directional von Mises-Fisher (vMF) feature likelihood ratio:
/// `exp(-kappa * (1.0 - cos_sim))`.
pub fn vmf_likelihood_ratio(f1: &[f64], f2: &[f64], kappa: f64) -> f64 {
    if f1.is_empty() || f2.is_empty() || f1.len() != f2.len() {
        return 1.0;
    }

    let mut dot = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for (&a, &b) in f1.iter().zip(f2.iter()) {
        dot += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }

    let mag = norm1.sqrt() * norm2.sqrt();
    if mag <= 1e-12 {
        return 1.0;
    }

    let cos_sim = (dot / mag).clamp(-1.0, 1.0);
    (-kappa * (1.0 - cos_sim)).exp()
}

/// Cosine similarity with precomputed L2 norms, rescaled to `[0.0, 1.0]`.
#[inline]
pub fn cosine_similarity_fast(f1: &[f64], norm1: f64, f2: &[f64], norm2: f64) -> f64 {
    if f1.is_empty() || f2.is_empty() || f1.len() != f2.len() {
        return 0.5;
    }

    let mag = norm1 * norm2;
    if mag <= 1e-12 {
        return 0.5;
    }

    let dot: f64 = f1.iter().zip(f2.iter()).map(|(&a, &b)| a * b).sum();
    let cos_sim = (dot / mag).clamp(-1.0, 1.0);
    0.5 * (cos_sim + 1.0)
}

/// Cosine similarity between two feature slices, rescaled to `[0.0, 1.0]`.
pub fn cosine_similarity(f1: &[f64], f2: &[f64]) -> f64 {
    if f1.is_empty() || f2.is_empty() || f1.len() != f2.len() {
        return 0.5; // neutral similarity
    }

    let mut dot = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for (&a, &b) in f1.iter().zip(f2.iter()) {
        dot += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }

    let mag = norm1.sqrt() * norm2.sqrt();
    if mag <= 1e-12 {
        return 0.5;
    }

    let cos_sim = dot / mag;
    let clamped = cos_sim.clamp(-1.0, 1.0);

    // Rescale from [-1, 1] to [0, 1]
    (clamped + 1.0) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_multivariate_erf_centered() {
        let mut covar = DMatrix::zeros(6, 6);
        covar[(0, 0)] = 4.0;
        covar[(1, 1)] = 4.0;
        covar[(2, 2)] = 4.0;

        let mu = DVector::from_vec(vec![10.0, 10.0, 10.0, 0.0, 0.0, 0.0]);
        let pred = Prediction::new(mu, covar);

        let pos = Vector3::new(10.0, 10.0, 10.0);
        let p = multivariate_erf(&pred, &pos, 2.0);

        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let f1 = vec![1.0, 0.0, 0.0];
        let f2 = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&f1, &f2) - 1.0).abs() < 1e-6);

        let f3 = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&f1, &f3) - 0.0).abs() < 1e-6);

        let f4 = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&f1, &f4) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_vmf_likelihood_ratio() {
        let f1 = vec![1.0, 0.0, 0.0];
        let f2 = vec![1.0, 0.0, 0.0]; // identical
        let f3 = vec![0.0, 1.0, 0.0]; // orthogonal
        let f4 = vec![-1.0, 0.0, 0.0]; // opposite
        let kappa = 3.0;

        let ratio_match = vmf_likelihood_ratio(&f1, &f2, kappa);
        assert!((ratio_match - 1.0).abs() < 1e-12);

        let ratio_ortho = vmf_likelihood_ratio(&f1, &f3, kappa);
        assert!((ratio_ortho - (-kappa).exp()).abs() < 1e-12);

        let ratio_opp = vmf_likelihood_ratio(&f1, &f4, kappa);
        assert!((ratio_opp - (-2.0 * kappa).exp()).abs() < 1e-12);

        // Missing features -> neutral (1.0)
        assert_eq!(vmf_likelihood_ratio(&[], &f1, kappa), 1.0);
    }
}
