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
/// Performs a numerically safe binary Bayesian update using the Law of Total Probability:
/// P(D) = P(D|A)P(A) + P(D|~A)P(~A)
///
/// Returns (update_multiplier, posterior).
#[pyfunction]
fn safe_bayesian_update(prior: f64, prob_assign: f64, prob_not_assign: f64) -> (f64, f64) {
    bayes::safe_bayesian_update(prior, prob_assign, prob_not_assign)
}

#[cfg(feature = "python")]
/// Computes cosine similarity between two feature vectors, rescaled to [0.0, 1.0].
#[pyfunction]
fn cosine_similarity(f1: Vec<f64>, f2: Vec<f64>) -> f64 {
    pdf::cosine_similarity(&f1, &f2)
}

#[cfg(feature = "python")]
/// Evaluates the directional von Mises-Fisher (vMF) feature likelihood ratio:
/// exp(-kappa * (1.0 - cos_sim))
///
/// Returns 1.0 for perfect matches or empty feature vectors, and decays exponentially as features diverge.
#[pyfunction]
fn vmf_likelihood_ratio(f1: Vec<f64>, f2: Vec<f64>, kappa: f64) -> f64 {
    pdf::vmf_likelihood_ratio(&f1, &f2, kappa)
}

#[cfg(feature = "python")]
fn extract_hypothesis(h: &Bound<'_, pyo3::PyAny>) -> PyResult<Hypothesis> {
    if let Ok(py_h) = h.extract::<PyRef<PyHypothesis>>() {
        return Ok(Hypothesis {
            hypothesis_type: py_h.hypothesis.into(),
            id: py_h.id,
            probability: py_h.probability,
            link_id: py_h.link_id,
            child_one_id: py_h.child_one_id,
            child_two_id: py_h.child_two_id,
            parent_one_id: py_h.parent_one_id,
            parent_two_id: py_h.parent_two_id,
        });
    }

    let hyp_val: u32 = if let Ok(val) = h.getattr("hypothesis") {
        if let Ok(v) = val.extract::<u32>() {
            v
        } else if let Ok(v) = val.getattr("value") {
            v.extract::<u32>()?
        } else {
            0
        }
    } else if let Ok(val) = h.getattr("hypothesis_type") {
        if let Ok(v) = val.getattr("value") {
            v.extract::<u32>()?
        } else {
            val.extract::<u32>()?
        }
    } else {
        0
    };

    let id: usize = if let Ok(v) = h.getattr("ID") {
        v.extract()?
    } else {
        h.getattr("id")?.extract()?
    };

    let prob: f64 = if let Ok(v) = h.getattr("probability") {
        v.extract()?
    } else {
        h.getattr("log_likelihood")?.extract()?
    };

    let link_id: usize = if let Ok(v) = h.getattr("link_ID") {
        v.extract().unwrap_or(0)
    } else if let Ok(v) = h.getattr("link_id") {
        v.extract().unwrap_or(0)
    } else {
        0
    };

    let child_one_id: usize = if let Ok(v) = h.getattr("child_one_ID") {
        v.extract().unwrap_or(0)
    } else if let Ok(v) = h.getattr("child_one_id") {
        v.extract().unwrap_or(0)
    } else {
        0
    };

    let child_two_id: usize = if let Ok(v) = h.getattr("child_two_ID") {
        v.extract().unwrap_or(0)
    } else if let Ok(v) = h.getattr("child_two_id") {
        v.extract().unwrap_or(0)
    } else {
        0
    };

    let parent_one_id: usize = if let Ok(v) = h.getattr("parent_one_ID") {
        v.extract().unwrap_or(0)
    } else if let Ok(v) = h.getattr("parent_one_id") {
        v.extract().unwrap_or(0)
    } else {
        0
    };

    let parent_two_id: usize = if let Ok(v) = h.getattr("parent_two_ID") {
        v.extract().unwrap_or(0)
    } else if let Ok(v) = h.getattr("parent_two_id") {
        v.extract().unwrap_or(0)
    } else {
        0
    };

    Ok(Hypothesis {
        hypothesis_type: hyp_val.into(),
        id,
        probability: prob,
        link_id,
        child_one_id,
        child_two_id,
        parent_one_id,
        parent_two_id,
    })
}

/// Standalone ILP optimization function for hypotheses.
#[cfg(feature = "python")]
#[pyfunction]
fn optimise_hypotheses(hyps: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<usize>> {
    let mut rust_hyps = Vec::new();
    for item in hyps.iter()? {
        let item = item?;
        rust_hyps.push(extract_hypothesis(&item)?);
    }

    let opt = TrackOptimiser::new(&rust_hyps);
    Ok(opt.optimise())
}

/// Python-compatible Hypothesis structure.
#[cfg(feature = "python")]
#[pyclass(name = "Hypothesis")]
#[derive(Debug, Clone)]
pub struct PyHypothesis {
    #[pyo3(get, set)]
    pub hypothesis: u32,
    #[pyo3(get, set, name = "ID")]
    pub id: usize,
    #[pyo3(get, set)]
    pub probability: f64,
    #[pyo3(get, set, name = "link_ID")]
    pub link_id: usize,
    #[pyo3(get, set, name = "child_one_ID")]
    pub child_one_id: usize,
    #[pyo3(get, set, name = "child_two_ID")]
    pub child_two_id: usize,
    #[pyo3(get, set, name = "parent_one_ID")]
    pub parent_one_id: usize,
    #[pyo3(get, set, name = "parent_two_ID")]
    pub parent_two_id: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyHypothesis {
    #[new]
    #[pyo3(signature = (hypothesis=0, id=0, probability=0.0, link_id=0, child_one_id=0, child_two_id=0, parent_one_id=0, parent_two_id=0))]
    fn new(
        hypothesis: u32,
        id: usize,
        probability: f64,
        link_id: usize,
        child_one_id: usize,
        child_two_id: usize,
        parent_one_id: usize,
        parent_two_id: usize,
    ) -> Self {
        Self {
            hypothesis,
            id,
            probability,
            link_id,
            child_one_id,
            child_two_id,
            parent_one_id,
            parent_two_id,
        }
    }

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

/// Statistics returned from tracking execution.
#[cfg(feature = "python")]
#[pyclass(name = "TrackingInfo")]
#[derive(Debug, Clone)]
pub struct PyTrackingInfo {
    #[pyo3(get)]
    pub error: u32,
    #[pyo3(get)]
    pub n_tracks: usize,
    #[pyo3(get)]
    pub n_active: usize,
    #[pyo3(get)]
    pub n_conflicts: usize,
    #[pyo3(get)]
    pub n_lost: usize,
    #[pyo3(get)]
    pub t_update_belief: f32,
    #[pyo3(get)]
    pub t_update_link: f32,
    #[pyo3(get)]
    pub t_total_time: f32,
    #[pyo3(get)]
    pub p_link: f32,
    #[pyo3(get)]
    pub p_lost: f32,
    #[pyo3(get)]
    pub complete: bool,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTrackingInfo {
    #[getter]
    fn tracker_active(&self) -> bool {
        !self.complete
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("error", self.error)?;
        dict.set_item("n_tracks", self.n_tracks)?;
        dict.set_item("n_active", self.n_active)?;
        dict.set_item("n_conflicts", self.n_conflicts)?;
        dict.set_item("n_lost", self.n_lost)?;
        dict.set_item("t_update_belief", self.t_update_belief)?;
        dict.set_item("t_update_link", self.t_update_link)?;
        dict.set_item("t_total_time", self.t_total_time)?;
        dict.set_item("p_link", self.p_link)?;
        dict.set_item("p_lost", self.p_lost)?;
        dict.set_item("complete", self.complete)?;
        Ok(dict)
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
    #[pyo3(signature = (measurements=3, states=6, a=None, h=None, p=None, q=None, r=None, dt=1.0, accuracy=2.0, max_lost=5, prob_not_assign=0.01))]
    fn new(
        measurements: usize,
        states: usize,
        a: Option<Vec<f64>>,
        h: Option<Vec<f64>>,
        p: Option<Vec<f64>>,
        q: Option<Vec<f64>>,
        r: Option<Vec<f64>>,
        dt: f64,
        accuracy: f64,
        max_lost: usize,
        prob_not_assign: f64,
    ) -> PyResult<Self> {
        let a_mat = match a {
            Some(v) => DMatrix::from_row_slice(states, states, &v),
            None => DMatrix::identity(states, states),
        };
        let h_mat = match h {
            Some(v) => DMatrix::from_row_slice(measurements, states, &v),
            None => {
                let mut m = DMatrix::zeros(measurements, states);
                for i in 0..measurements.min(states) {
                    m[(i, i)] = 1.0;
                }
                m
            }
        };
        let p_mat = match p {
            Some(v) => DMatrix::from_row_slice(states, states, &v),
            None => DMatrix::identity(states, states) * 10.0,
        };
        let q_mat = match q {
            Some(v) => DMatrix::from_row_slice(states, states, &v),
            None => DMatrix::identity(states, states) * 0.1,
        };
        let r_mat = match r {
            Some(v) => DMatrix::from_row_slice(measurements, measurements, &v),
            None => DMatrix::identity(measurements, measurements) * 0.1,
        };

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

    fn set_motion_model(
        &mut self,
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
    ) {
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

        self.inner.motion_model = motion_model;
        self.inner.config.accuracy = accuracy;
        self.inner.config.max_lost = max_lost;
        self.inner.config.prob_not_assign = prob_not_assign;
    }

    fn set_update_mode(&mut self, mode: usize) {
        self.inner.config.update_mode = mode;
    }

    fn set_update_features(&mut self, features: u32) {
        self.inner.config.use_motion_features = (features & 1) != 0;
        self.inner.config.use_visual_features = (features & 2) != 0;
    }

    fn set_feature_concentration(&mut self, kappa: f64) {
        self.inner.config.feature_concentration = kappa;
    }

    fn get_feature_concentration(&self) -> f64 {
        self.inner.config.feature_concentration
    }

    fn set_max_search_radius(&mut self, radius: f64) {
        self.inner.config.max_search_radius = radius;
    }

    fn set_store_candidate_graph(&mut self, store: bool) {
        self.inner.set_store_candidate_graph(store);
    }

    fn set_volume(&mut self, bounds: Vec<f64>) {
        if bounds.len() >= 6 {
            let b = [bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]];
            self.inner.volume.set_volume(&b);
        }
    }

    fn get_volume(&self) -> Vec<f64> {
        vec![
            self.inner.volume.min_xyz[0],
            self.inner.volume.max_xyz[0],
            self.inner.volume.min_xyz[1],
            self.inner.volume.max_xyz[1],
            self.inner.volume.min_xyz[2],
            self.inner.volume.max_xyz[2],
        ]
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

    #[pyo3(name = "append")]
    fn append_py(&mut self, obj: &Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        let id: i64 = obj.getattr("ID")?.extract()?;
        let x: f64 = obj.getattr("x")?.extract()?;
        let y: f64 = obj.getattr("y")?.extract()?;
        let z: f64 = obj.getattr("z")?.extract()?;
        let t: usize = obj.getattr("t")?.extract()?;
        let dummy: bool = obj.getattr("dummy").and_then(|v| v.extract()).unwrap_or(false);
        let states: usize = obj.getattr("states").and_then(|v| v.extract()).unwrap_or(1);
        let label: u32 = obj.getattr("label").and_then(|v| v.extract()).unwrap_or(0);

        let n_features: usize = obj.getattr("n_features").and_then(|v| v.extract()).unwrap_or(0);
        let feat = if n_features > 0 {
            if let Ok(feat_obj) = obj.getattr("_features") {
                if let Ok(v) = feat_obj.extract::<Vec<f64>>() {
                    DVector::from_vec(v)
                } else {
                    DVector::zeros(0)
                }
            } else {
                DVector::zeros(0)
            }
        } else {
            DVector::zeros(0)
        };

        let track_obj = TrackObject {
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
        self.inner.append(vec![track_obj]);
        Ok(())
    }

    #[pyo3(signature = (n_steps=None))]
    fn step(&mut self, n_steps: Option<usize>) -> PyResult<PyTrackingInfo> {
        let n = n_steps.unwrap_or(1);
        let info = self.inner.step(n).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Tracking step failed with error {:?}", e))
        })?;

        Ok(PyTrackingInfo {
            error: info.error as u32,
            n_tracks: info.n_tracks,
            n_active: info.n_active,
            n_conflicts: info.n_conflicts,
            n_lost: info.n_lost,
            t_update_belief: info.t_update_belief,
            t_update_link: info.t_update_link,
            t_total_time: info.t_total_time,
            p_link: info.p_link,
            p_lost: info.p_lost,
            complete: info.complete,
        })
    }

    fn track(&mut self) -> PyResult<PyTrackingInfo> {
        let info = self.inner.track_all().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Tracking failed with error {:?}", e))
        })?;

        Ok(PyTrackingInfo {
            error: info.error as u32,
            n_tracks: info.n_tracks,
            n_active: info.n_active,
            n_conflicts: info.n_conflicts,
            n_lost: info.n_lost,
            t_update_belief: info.t_update_belief,
            t_update_link: info.t_update_link,
            t_total_time: info.t_total_time,
            p_link: info.p_link,
            p_lost: info.p_lost,
            complete: info.complete,
        })
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

    #[pyo3(name = "get_ID")]
    fn get_id_alias(&self, trk_idx: usize) -> usize {
        self.get_id(trk_idx)
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
        PyArray2::from_vec2_bound(
            py,
            &data.chunks(3).map(|c| c.to_vec()).collect::<Vec<_>>(),
        ).unwrap()
    }

    fn get_kalman_mu<'py>(&self, py: Python<'py>, trk_idx: usize) -> Bound<'py, PyArray2<f64>> {
        let trk = &self.inner.manager.tracks[trk_idx];
        let mut rows = Vec::new();
        for i in 0..trk.length() {
            let t = trk.track[i].t as f64;
            let mu = &trk.kalman[i].mu;
            rows.push(vec![t, mu[0], mu[1], mu[2]]);
        }
        PyArray2::from_vec2_bound(py, &rows).unwrap()
    }

    fn get_kalman_covar<'py>(&self, py: Python<'py>, trk_idx: usize) -> Bound<'py, PyArray2<f64>> {
        let trk = &self.inner.manager.tracks[trk_idx];
        let mut rows = Vec::new();
        for i in 0..trk.length() {
            let t = trk.track[i].t as f64;
            let c = &trk.kalman[i].covar;
            rows.push(vec![
                t,
                c[(0, 0)], c[(0, 1)], c[(0, 2)],
                c[(1, 0)], c[(1, 1)], c[(1, 2)],
                c[(2, 0)], c[(2, 1)], c[(2, 2)],
            ]);
        }
        PyArray2::from_vec2_bound(py, &rows).unwrap()
    }

    fn get_kalman_pred<'py>(&self, py: Python<'py>, trk_idx: usize) -> Bound<'py, PyArray2<f64>> {
        let trk = &self.inner.manager.tracks[trk_idx];
        let mut rows = Vec::new();
        for i in 0..trk.length() {
            let t = trk.track[i].t as f64;
            let mu = &trk.prediction[i].mu;
            rows.push(vec![t, mu[0], mu[1], mu[2]]);
        }
        PyArray2::from_vec2_bound(py, &rows).unwrap()
    }

    fn get_label<'py>(&self, py: Python<'py>, trk_idx: usize) -> Bound<'py, PyArray2<u32>> {
        let trk = &self.inner.manager.tracks[trk_idx];
        let mut rows = Vec::new();
        for i in 0..trk.length() {
            let t = trk.track[i].t as u32;
            let lbl = trk.track[i].label;
            rows.push(vec![t, lbl]);
        }
        PyArray2::from_vec2_bound(py, &rows).unwrap()
    }

    fn get_dummy(&self, dummy_id: i64) -> Option<(i64, f64, f64, f64, usize, u32, bool)> {
        self.inner
            .manager
            .dummies
            .iter()
            .find(|d| d.id == dummy_id)
            .map(|d| (d.id, d.x, d.y, d.z, d.t, d.label, d.dummy))
    }

    fn num_edges(&self) -> usize {
        self.inner.manager.num_edges()
    }

    fn get_graph_edge(&self, idx: usize) -> Option<(i64, i64, f64, u32)> {
        self.inner
            .manager
            .graph_edges
            .get(idx)
            .map(|e| (e.source, e.target, e.score, e.edge_type))
    }

    #[pyo3(signature = (params=None, start_frame=None, stop_frame=None))]
    fn create_hypotheses(
        &mut self,
        params: Option<&Bound<'_, pyo3::PyAny>>,
        start_frame: Option<usize>,
        stop_frame: Option<usize>,
    ) -> usize {
        let mut hp = HypothesisParams::default();

        if let Some(p) = params {
            if let Ok(dict) = p.downcast::<pyo3::types::PyDict>() {
                if let Ok(Some(v)) = dict.get_item("lambda_time") { if let Ok(val) = v.extract() { hp.lambda_time = val; } }
                if let Ok(Some(v)) = dict.get_item("lambda_dist") { if let Ok(val) = v.extract() { hp.lambda_dist = val; } }
                if let Ok(Some(v)) = dict.get_item("lambda_link") { if let Ok(val) = v.extract() { hp.lambda_link = val; } }
                if let Ok(Some(v)) = dict.get_item("lambda_branch") { if let Ok(val) = v.extract() { hp.lambda_branch = val; } }
                if let Ok(Some(v)) = dict.get_item("eta") { if let Ok(val) = v.extract() { hp.eta = val; } }
                if let Ok(Some(v)) = dict.get_item("theta_dist") { if let Ok(val) = v.extract() { hp.theta_dist = val; } }
                if let Ok(Some(v)) = dict.get_item("theta_time") { if let Ok(val) = v.extract() { hp.theta_time = val; } }
                if let Ok(Some(v)) = dict.get_item("dist_thresh") { if let Ok(val) = v.extract() { hp.dist_thresh = val; } }
                if let Ok(Some(v)) = dict.get_item("time_thresh") { if let Ok(val) = v.extract() { hp.time_thresh = val; } }
                if let Ok(Some(v)) = dict.get_item("apop_thresh") { if let Ok(val) = v.extract() { hp.apop_thresh = val; } }
                if let Ok(Some(v)) = dict.get_item("segmentation_miss_rate") { if let Ok(val) = v.extract() { hp.segmentation_miss_rate = val; } }
                if let Ok(Some(v)) = dict.get_item("apoptosis_rate") { if let Ok(val) = v.extract() { hp.apoptosis_rate = val; } }
                if let Ok(Some(v)) = dict.get_item("relax") { if let Ok(val) = v.extract() { hp.relax = val; } }
                if let Ok(Some(v)) = dict.get_item("hypotheses_to_generate") { if let Ok(val) = v.extract() { hp.hypotheses_to_generate = val; } }
            } else {
                if let Ok(v) = p.getattr("lambda_time") { if let Ok(val) = v.extract() { hp.lambda_time = val; } }
                if let Ok(v) = p.getattr("lambda_dist") { if let Ok(val) = v.extract() { hp.lambda_dist = val; } }
                if let Ok(v) = p.getattr("lambda_link") { if let Ok(val) = v.extract() { hp.lambda_link = val; } }
                if let Ok(v) = p.getattr("lambda_branch") { if let Ok(val) = v.extract() { hp.lambda_branch = val; } }
                if let Ok(v) = p.getattr("eta") { if let Ok(val) = v.extract() { hp.eta = val; } }
                if let Ok(v) = p.getattr("theta_dist") { if let Ok(val) = v.extract() { hp.theta_dist = val; } }
                if let Ok(v) = p.getattr("theta_time") { if let Ok(val) = v.extract() { hp.theta_time = val; } }
                if let Ok(v) = p.getattr("dist_thresh") { if let Ok(val) = v.extract() { hp.dist_thresh = val; } }
                if let Ok(v) = p.getattr("time_thresh") { if let Ok(val) = v.extract() { hp.time_thresh = val; } }
                if let Ok(v) = p.getattr("apop_thresh") { if let Ok(val) = v.extract() { hp.apop_thresh = val; } }
                if let Ok(v) = p.getattr("segmentation_miss_rate") { if let Ok(val) = v.extract() { hp.segmentation_miss_rate = val; } }
                if let Ok(v) = p.getattr("apoptosis_rate") { if let Ok(val) = v.extract() { hp.apoptosis_rate = val; } }
                if let Ok(v) = p.getattr("relax") { if let Ok(val) = v.extract() { hp.relax = val; } }
                if let Ok(v) = p.getattr("hypotheses_to_generate") { if let Ok(val) = v.extract() { hp.hypotheses_to_generate = val; } }
            }
        }

        let f0 = start_frame.unwrap_or(0);
        let f1 = stop_frame.unwrap_or(usize::MAX);

        let mut engine = HypothesisEngine::new(hp, &self.inner.volume, (f0, f1));
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

    fn merge(&mut self, selected_indices: Vec<usize>) {
        let selected: Vec<Hypothesis> = selected_indices
            .into_iter()
            .filter_map(|idx| self.stored_hypotheses.get(idx).cloned())
            .collect();
        self.inner.manager.merge(&selected);
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn _btrack_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(safe_bayesian_update, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(vmf_likelihood_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(optimise_hypotheses, m)?)?;
    m.add_class::<PyBayesianTracker>()?;
    m.add_class::<PyHypothesis>()?;
    m.add_class::<PyTrackingInfo>()?;
    Ok(())
}
