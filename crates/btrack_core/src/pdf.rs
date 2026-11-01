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
}
