//! btrack_core: Fast Bayesian multi-object tracking core in Rust.

pub mod types;
pub mod motion;
pub mod bayes;
pub mod pdf;
pub mod hyperbin;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction]
fn safe_bayesian_update(prior: f64, prob_assign: f64, prob_not_assign: f64) -> (f64, f64) {
    bayes::safe_bayesian_update(prior, prob_assign, prob_not_assign)
}

#[cfg(feature = "python")]
#[pyfunction]
fn cosine_similarity(f1: Vec<f64>, f2: Vec<f64>) -> f64 {
    pdf::cosine_similarity(&f1, &f2)
}

#[cfg(feature = "python")]
#[pymodule]
fn _btrack_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(safe_bayesian_update, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    Ok(())
}
