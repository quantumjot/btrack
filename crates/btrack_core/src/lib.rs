//! btrack_core: Fast Bayesian multi-object tracking core in Rust.

pub mod types;
pub mod motion;
pub mod bayes;
pub mod pdf;
pub mod hyperbin;
pub mod tracklet;
pub mod manager;
pub mod tracker;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::PyArray2;
#[cfg(feature = "python")]
use nalgebra::{DMatrix, DVector};

#[cfg(feature = "python")]
use crate::tracker::{BayesianTracker, TrackerConfig};
#[cfg(feature = "python")]
use crate::motion::MotionModel;
#[cfg(feature = "python")]
use crate::types::TrackObject;

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

/// PyO3 Python wrapper around BayesianTracker.
#[cfg(feature = "python")]
#[pyclass(name = "BayesianTracker")]
pub struct PyBayesianTracker {
    inner: BayesianTracker,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBayesianTracker {
    #[new]
    #[pyo3(signature = (measurements, states, a, h, p, q, r, dt=1.0, accuracy=2.0, max_lost=5, prob_not_assign=0.01))]
    fn new(
        measurements: usize,
        states: usize,
        a: Vec<f64>,
        h: Vec<f64>,
        p: Vec<f64>,
        q: Vec<f64>,
        r: Vec<f64>,
        dt: f64,
        accuracy: f64,
        max_lost: usize,
        prob_not_assign: f64,
    ) -> PyResult<Self> {
        let a_mat = DMatrix::from_row_slice(states, states, &a);
        let h_mat = DMatrix::from_row_slice(measurements, states, &h);
        let p_mat = DMatrix::from_row_slice(states, states, &p);
        let q_mat = DMatrix::from_row_slice(states, states, &q);
        let r_mat = DMatrix::from_row_slice(measurements, measurements, &r);

        let mut motion_model = MotionModel::new(a_mat, h_mat, p_mat, r_mat, q_mat);
        motion_model.dt = dt;
        motion_model.accuracy = accuracy;
        motion_model.max_lost = max_lost;
        motion_model.prob_not_assign = prob_not_assign;

        let mut config = TrackerConfig::default();
        config.accuracy = accuracy;
        config.max_lost = max_lost;
        config.prob_not_assign = prob_not_assign;

        Ok(Self {
            inner: BayesianTracker::new(motion_model, config),
        })
    }

    fn set_update_mode(&mut self, mode: usize) {
        self.inner.config.update_mode = mode;
    }

    fn set_update_features(&mut self, features: u32) {
        self.inner.config.use_motion_features = (features & 1) != 0;
        self.inner.config.use_visual_features = (features & 2) != 0;
    }

    fn set_max_search_radius(&mut self, radius: f64) {
        self.inner.config.max_search_radius = radius;
    }

    fn set_store_candidate_graph(&mut self, store: bool) {
        self.inner.set_store_candidate_graph(store);
    }

    #[pyo3(signature = (id, x, y, z, t, dummy, states, label, features=None))]
    fn append_object(
        &mut self,
        id: i64,
        x: f64,
        y: f64,
        z: f64,
        t: usize,
        dummy: bool,
        states: usize,
        label: u32,
        features: Option<Vec<f64>>,
    ) {
        let feat = match features {
            Some(f) => DVector::from_vec(f),
            None => DVector::zeros(0),
        };

        let obj = TrackObject {
            id,
            x,
            y,
            z,
            t,
            dummy,
            states,
            label,
            features: feat,
        };

        self.inner.append(vec![obj]);
    }

    fn step(&mut self, n_steps: usize) -> PyResult<bool> {
        let info = self.inner.step(n_steps).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Tracking step failed with error {:?}", e))
        })?;
        Ok(info.complete)
    }

    fn track(&mut self) -> PyResult<()> {
        self.inner.track_all().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Tracking failed with error {:?}", e))
        })?;
        Ok(())
    }

    fn size(&self) -> usize {
        self.inner.manager.num_tracks()
    }

    fn track_length(&self, trk_idx: usize) -> usize {
        if trk_idx < self.inner.manager.tracks.len() {
            self.inner.manager.tracks[trk_idx].length()
        } else {
            0
        }
    }

    fn get_id(&self, trk_idx: usize) -> usize {
        self.inner.manager.tracks[trk_idx].id
    }

    fn get_parent(&self, trk_idx: usize) -> usize {
        self.inner.manager.tracks[trk_idx].parent
    }

    fn get_root(&self, trk_idx: usize) -> usize {
        self.inner.manager.tracks[trk_idx].root
    }

    fn get_children(&self, trk_idx: usize) -> (usize, usize) {
        let trk = &self.inner.manager.tracks[trk_idx];
        (trk.child_one, trk.child_two)
    }

    fn get_generation(&self, trk_idx: usize) -> usize {
        self.inner.manager.tracks[trk_idx].generation
    }

    fn get_fate(&self, trk_idx: usize) -> u32 {
        self.inner.manager.tracks[trk_idx].fate as u32
    }

    fn get_refs(&self, trk_idx: usize) -> Vec<i64> {
        self.inner.manager.tracks[trk_idx]
            .track
            .iter()
            .map(|o| o.id)
            .collect()
    }

    fn get_positions<'py>(&self, py: Python<'py>, trk_idx: usize) -> Bound<'py, PyArray2<f64>> {
        let trk = &self.inner.manager.tracks[trk_idx];
        let n = trk.length();
        let mut data = Vec::with_capacity(n * 3);
        for obj in &trk.track {
            data.push(obj.x);
            data.push(obj.y);
            data.push(obj.z);
        }
        let arr = PyArray2::from_vec2_bound(
            py,
            &data.chunks(3).map(|c| c.to_vec()).collect::<Vec<_>>(),
        ).unwrap();
        arr
    }

    fn get_dummy(&self, dummy_id: i64) -> Option<(i64, f64, f64, f64, usize)> {
        self.inner
            .manager
            .dummies
            .iter()
            .find(|d| d.id == dummy_id)
            .map(|d| (d.id, d.x, d.y, d.z, d.t))
    }

    fn num_edges(&self) -> usize {
        self.inner.manager.num_edges()
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn _btrack_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(safe_bayesian_update, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_class::<PyBayesianTracker>()?;
    Ok(())
}
