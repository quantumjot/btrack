//! btrack_core: Fast Bayesian multi-object tracking core in Rust.

pub mod types;
pub mod motion;
pub mod bayes;
pub mod pdf;
pub mod hyperbin;
pub mod tracklet;
pub mod manager;
pub mod tracker;
pub mod hypothesis;
pub mod optimiser;

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
use crate::hypothesis::{Hypothesis, HypothesisEngine, HypothesisParams};
#[cfg(feature = "python")]
use crate::optimiser::TrackOptimiser;

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

/// Python-compatible Hypothesis structure.
#[cfg(feature = "python")]
#[pyclass(name = "Hypothesis")]
#[derive(Debug, Clone)]
pub struct PyHypothesis {
    #[pyo3(get)]
    pub hypothesis: u32,
    #[pyo3(get, name = "ID")]
    pub id: usize,
    #[pyo3(get)]
    pub probability: f64,
    #[pyo3(get, name = "link_ID")]
    pub link_id: usize,
    #[pyo3(get, name = "child_one_ID")]
    pub child_one_id: usize,
    #[pyo3(get, name = "child_two_ID")]
    pub child_two_id: usize,
    #[pyo3(get, name = "parent_one_ID")]
    pub parent_one_id: usize,
    #[pyo3(get, name = "parent_two_ID")]
    pub parent_two_id: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyHypothesis {
    #[getter]
    fn hypothesis_type(&self) -> u32 {
        self.hypothesis
    }

    #[getter]
    fn log_likelihood(&self) -> f64 {
        self.probability
    }

    fn __repr__(&self) -> String {
        format!(
            "Hypothesis(type={}, ID={}, prob={:.4}, link={}, children=({}, {}))",
            self.hypothesis, self.id, self.probability, self.link_id, self.child_one_id, self.child_two_id
        )
    }
}

/// PyO3 Python wrapper around BayesianTracker.
#[cfg(feature = "python")]
#[pyclass(name = "BayesianTracker")]
pub struct PyBayesianTracker {
    inner: BayesianTracker,
    stored_hypotheses: Vec<Hypothesis>,
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
            stored_hypotheses: Vec::new(),
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

    #[pyo3(signature = (params_dict=None, start_frame=None, stop_frame=None))]
    fn create_hypotheses(
        &mut self,
        params_dict: Option<&Bound<'_, pyo3::types::PyDict>>,
        start_frame: Option<usize>,
        stop_frame: Option<usize>,
    ) -> usize {
        let mut params = HypothesisParams::default();

        if let Some(dict) = params_dict {
            if let Ok(Some(v)) = dict.get_item("lambda_time") {
                if let Ok(val) = v.extract() { params.lambda_time = val; }
            }
            if let Ok(Some(v)) = dict.get_item("lambda_dist") {
                if let Ok(val) = v.extract() { params.lambda_dist = val; }
            }
            if let Ok(Some(v)) = dict.get_item("lambda_link") {
                if let Ok(val) = v.extract() { params.lambda_link = val; }
            }
            if let Ok(Some(v)) = dict.get_item("lambda_branch") {
                if let Ok(val) = v.extract() { params.lambda_branch = val; }
            }
            if let Ok(Some(v)) = dict.get_item("eta") {
                if let Ok(val) = v.extract() { params.eta = val; }
            }
            if let Ok(Some(v)) = dict.get_item("theta_dist") {
                if let Ok(val) = v.extract() { params.theta_dist = val; }
            }
            if let Ok(Some(v)) = dict.get_item("theta_time") {
                if let Ok(val) = v.extract() { params.theta_time = val; }
            }
            if let Ok(Some(v)) = dict.get_item("dist_thresh") {
                if let Ok(val) = v.extract() { params.dist_thresh = val; }
            }
            if let Ok(Some(v)) = dict.get_item("time_thresh") {
                if let Ok(val) = v.extract() { params.time_thresh = val; }
            }
            if let Ok(Some(v)) = dict.get_item("apop_thresh") {
                if let Ok(val) = v.extract() { params.apop_thresh = val; }
            }
            if let Ok(Some(v)) = dict.get_item("segmentation_miss_rate") {
                if let Ok(val) = v.extract() { params.segmentation_miss_rate = val; }
            }
            if let Ok(Some(v)) = dict.get_item("apoptosis_rate") {
                if let Ok(val) = v.extract() { params.apoptosis_rate = val; }
            }
            if let Ok(Some(v)) = dict.get_item("relax") {
                if let Ok(val) = v.extract() { params.relax = val; }
            }
            if let Ok(Some(v)) = dict.get_item("hypotheses_to_generate") {
                if let Ok(val) = v.extract() { params.hypotheses_to_generate = val; }
            }
        }

        let f0 = start_frame.unwrap_or(0);
        let f1 = stop_frame.unwrap_or(usize::MAX);

        let mut engine = HypothesisEngine::new(params, &self.inner.volume, (f0, f1));
        let hyps = engine.create(&self.inner.manager);
        let count = hyps.len();
        self.stored_hypotheses = hyps;
        count
    }

    fn get_hypotheses(&self) -> Vec<PyHypothesis> {
        self.stored_hypotheses
            .iter()
            .map(|h| PyHypothesis {
                hypothesis: h.hypothesis_type as u32,
                id: h.id,
                probability: h.probability,
                link_id: h.link_id,
                child_one_id: h.child_one_id,
                child_two_id: h.child_two_id,
                parent_one_id: h.parent_one_id,
                parent_two_id: h.parent_two_id,
            })
            .collect()
    }

    fn get_hypothesis(&self, idx: usize) -> Option<PyHypothesis> {
        self.stored_hypotheses.get(idx).map(|h| PyHypothesis {
            hypothesis: h.hypothesis_type as u32,
            id: h.id,
            probability: h.probability,
            link_id: h.link_id,
            child_one_id: h.child_one_id,
            child_two_id: h.child_two_id,
            parent_one_id: h.parent_one_id,
            parent_two_id: h.parent_two_id,
        })
    }

    fn optimise(&self) -> Vec<usize> {
        let opt = TrackOptimiser::new(&self.stored_hypotheses);
        opt.optimise()
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn _btrack_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(safe_bayesian_update, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_class::<PyBayesianTracker>()?;
    m.add_class::<PyHypothesis>()?;
    Ok(())
}
