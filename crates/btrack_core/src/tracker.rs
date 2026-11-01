//! Bayesian multi-object tracking engine.

use std::collections::{HashMap, HashSet};
use nalgebra::DMatrix;

use crate::bayes::safe_bayesian_update;
use crate::hyperbin::ObjectBin;
use crate::manager::TrackManager;
use crate::motion::MotionModel;
use crate::pdf::{cosine_similarity, multivariate_erf};
use crate::tracklet::Tracklet;
use crate::types::{
    ImagingVolume, ObjectStateLabel, TrackObject, TrackingError, TrackingInfo,
    DEFAULT_ACCURACY, MAX_LOST, MAX_SEARCH_RADIUS, PROB_NOT_ASSIGN,
};

/// Configuration parameters for the BayesianTracker.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    pub max_search_radius: f64,
    pub accuracy: f64,
    pub prob_not_assign: f64,
    pub max_lost: usize,
    pub use_motion_features: bool,
    pub use_visual_features: bool,
    pub update_mode: usize, // 0 = Exact, 1 = Approximate
    pub disallow_metaphase_anaphase_linking: bool,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_search_radius: MAX_SEARCH_RADIUS,
            accuracy: DEFAULT_ACCURACY,
            prob_not_assign: PROB_NOT_ASSIGN,
            max_lost: MAX_LOST,
            use_motion_features: true,
            use_visual_features: false,
            update_mode: 0,
            disallow_metaphase_anaphase_linking: true,
        }
    }
}

/// Bayesian multi-object tracker coordinating state estimation, belief matrix computation, and track linkage.
#[derive(Debug, Clone)]
pub struct BayesianTracker {
    pub manager: TrackManager,
    pub volume: ImagingVolume,
    pub motion_model: MotionModel,
    pub config: TrackerConfig,
    pub statistics: TrackingInfo,

    frames: Vec<usize>,
    current_frame: usize,
    active_indices: Vec<usize>,
    initialised: bool,
    track_id_counter: usize,
    node_cursor: usize,
}

impl BayesianTracker {
    pub fn new(motion_model: MotionModel, config: TrackerConfig) -> Self {
        Self {
            manager: TrackManager::new(),
            volume: ImagingVolume::default(),
            motion_model,
            config,
            statistics: TrackingInfo::default(),
            frames: Vec::new(),
            current_frame: 0,
            active_indices: Vec::new(),
            initialised: false,
            track_id_counter: 1,
            node_cursor: 0,
        }
    }

    pub fn append(&mut self, objects: Vec<TrackObject>) {
        for obj in objects {
            self.volume.update(&obj.position());
            self.manager.push_node(obj);
        }
        self.initialised = false;
    }

    pub fn set_store_candidate_graph(&mut self, store: bool) {
        self.manager.store_candidate_graph = store;
    }

    fn next_track_id(&mut self) -> usize {
        let id = self.track_id_counter;
        self.track_id_counter += 1;
        id
    }

    /// Initializes tracking from the first frame of observations.
    pub fn initialise(&mut self) -> Result<(), TrackingError> {
        if self.manager.num_nodes() == 0 {
            self.statistics.error = TrackingError::EmptyQueue;
            return Err(TrackingError::EmptyQueue);
        }

        self.manager.sort_nodes();

        // Extract sorted unique frame numbers
        let mut frames: Vec<usize> = self.manager.graph_nodes.iter().map(|o| o.t).collect();
        frames.sort_unstable();
        frames.dedup();

        if frames.is_empty() {
            self.statistics.error = TrackingError::NoTracks;
            return Err(TrackingError::NoTracks);
        }

        // Validate frame continuity against max_lost
        let mut usable = false;
        for i in 1..frames.len() {
            if (frames[i] - frames[i - 1]) <= self.config.max_lost {
                usable = true;
                break;
            }
        }
        if frames.len() == 1 {
            usable = true;
        }

        if !usable {
            self.statistics.error = TrackingError::NoUseableFrames;
            return Err(TrackingError::NoUseableFrames);
        }

        self.frames = frames;
        self.current_frame = self.frames[0];
        self.node_cursor = 0;
        self.track_id_counter = 1;

        // Initialize first tracklets from objects in the first frame
        while self.node_cursor < self.manager.num_nodes()
            && self.manager.graph_nodes[self.node_cursor].t == self.current_frame
        {
            let obj = self.manager.graph_nodes[self.node_cursor].clone();
            let trk = Tracklet::new(
                self.next_track_id(),
                obj,
                self.config.max_lost,
                self.motion_model.clone(),
            );
            self.manager.push_track(trk);
            self.node_cursor += 1;
        }

        self.current_frame += 1;
        self.initialised = true;
        self.statistics.error = TrackingError::Success;
        Ok(())
    }

    fn update_active(&mut self) {
        self.active_indices.clear();

        for i in 0..self.manager.tracks.len() {
            let trk = &mut self.manager.tracks[i];
            if !self.volume.inside(&trk.position()) {
                trk.set_lost();
                continue;
            }

            if trk.active() {
                self.active_indices.push(i);
            } else {
                trk.trim();
            }
        }
    }

    /// Advances the tracker by `n_steps` frames.
    pub fn step(&mut self, n_steps: usize) -> Result<TrackingInfo, TrackingError> {
        if n_steps == 0 {
            return Ok(self.statistics.clone());
        }

        if !self.initialised {
            self.initialise()?;
        }

        let mut step = 0;
        let last_frame = *self.frames.last().unwrap_or(&0);

        while step < n_steps && self.current_frame <= last_frame {
            self.update_active();

            // Collect observations in current frame
            let mut new_objects = Vec::new();
            while self.node_cursor < self.manager.num_nodes()
                && self.manager.graph_nodes[self.node_cursor].t == self.current_frame
            {
                new_objects.push(self.manager.graph_nodes[self.node_cursor].clone());
                self.node_cursor += 1;
            }

            let n_active = self.active_indices.len();
            let n_obs = new_objects.len();

            // Handle empty frame
            if new_objects.is_empty() {
                for &idx in &self.active_indices {
                    self.manager.tracks[idx].append_dummy(self.config.use_motion_features);
                }
                step += 1;
                self.current_frame += 1;
                continue;
            }

            // Compute belief matrix: rows = (n_obs + 1), cols = n_active
            let mut belief = DMatrix::zeros(n_obs + 1, n_active);

            if self.config.update_mode == 1 {
                self.cost_approximate(&mut belief, &new_objects);
            } else {
                self.cost_exact(&mut belief, &new_objects);
            }

            // Store candidate graph edges if enabled
            if self.manager.store_candidate_graph {
                for (col, &trk_idx) in self.active_indices.iter().enumerate() {
                    let last_id = self.manager.tracks[trk_idx].track.last().unwrap().id;
                    for (row, new_obj) in new_objects.iter().enumerate() {
                        let score = belief[(row, col)];
                        self.manager.push_edge(last_id, new_obj.id, score, 0);
                    }
                }
            }

            // Link tracks with observations based on posterior belief
            self.link(&belief, &new_objects);

            step += 1;
            self.current_frame += 1;
        }

        if self.current_frame > last_frame {
            self.statistics.complete = true;
            self.manager.finalise();
        }

        self.statistics.n_tracks = self.manager.num_tracks();
        self.statistics.n_active = self.active_indices.len();
        Ok(self.statistics.clone())
    }

    /// Computes full exact belief matrix for all active tracks and candidate observations.
    fn cost_exact(&self, belief: &mut DMatrix<f64>, new_objects: &[TrackObject]) {
        let n_obs = new_objects.len();
        let uniform_prior = 1.0 / (n_obs as f64 + 1.0);

        for (col, &trk_idx) in self.active_indices.iter().enumerate() {
            let trk = &self.manager.tracks[trk_idx];
            let mut col_prob = DMatrix::from_element(n_obs + 1, 1, uniform_prior);

            // Pass 1: Motion Model / Spatial Likelihood
            if self.config.use_motion_features {
                let pred = trk.predict();
                let last_label = trk.track.last().map(|o| o.label).unwrap_or(0);

                for (row, obj) in new_objects.iter().enumerate() {
                    let mut prob_assign = multivariate_erf(&pred, &obj.position(), self.config.accuracy);

                    if self.config.disallow_metaphase_anaphase_linking
                        && last_label == ObjectStateLabel::Metaphase as u32
                        && obj.label == ObjectStateLabel::Anaphase as u32
                    {
                        prob_assign = 0.0;
                    }

                    let prior_assign = col_prob[(row, 0)];
                    let (safe_update, posterior) = safe_bayesian_update(
                        prior_assign,
                        prob_assign,
                        self.config.prob_not_assign,
                    );

                    for r in 0..=n_obs {
                        if r != row {
                            col_prob[(r, 0)] *= safe_update;
                        }
                    }
                    col_prob[(row, 0)] = posterior;
                }
            }

            // Pass 2: Visual Features
            if self.config.use_visual_features {
                let last_features = &trk.track.last().unwrap().features;

                for (row, obj) in new_objects.iter().enumerate() {
                    let prob_assign = cosine_similarity(last_features.as_slice(), obj.features.as_slice());
                    let prior_assign = col_prob[(row, 0)];
                    let (safe_update, posterior) = safe_bayesian_update(
                        prior_assign,
                        prob_assign,
                        self.config.prob_not_assign,
                    );

                    for r in 0..=n_obs {
                        if r != row {
                            col_prob[(r, 0)] *= safe_update;
                        }
                    }
                    col_prob[(row, 0)] = posterior;
                }
            }

            for row in 0..=n_obs {
                belief[(row, col)] = col_prob[(row, 0)];
            }
        }
    }

    /// Computes approximate belief matrix restricted to neighboring spatial bins.
    fn cost_approximate(&self, belief: &mut DMatrix<f64>, new_objects: &[TrackObject]) {
        let n_obs = new_objects.len();
        let mut bin = ObjectBin::new(self.config.max_search_radius, 1.0);
        for (i, obj) in new_objects.iter().enumerate() {
            bin.add_object(obj.clone(), i);
        }

        for (col, &trk_idx) in self.active_indices.iter().enumerate() {
            let trk = &self.manager.tracks[trk_idx];
            let neighbors = bin.get_near(&trk.position());

            if neighbors.is_empty() {
                // If no objects within radius, track is lost
                for r in 0..n_obs {
                    belief[(r, col)] = 0.0;
                }
                belief[(n_obs, col)] = 1.0;
                continue;
            }

            let local_prior = 1.0 / (neighbors.len() as f64 + 1.0);
            let mut col_prob = DMatrix::zeros(n_obs + 1, 1);
            for (_, idx) in &neighbors {
                col_prob[(*idx, 0)] = local_prior;
            }
            col_prob[(n_obs, 0)] = local_prior;

            if self.config.use_motion_features {
                let pred = trk.predict();
                let last_label = trk.track.last().map(|o| o.label).unwrap_or(0);

                for (obj, idx) in &neighbors {
                    let mut prob_assign = multivariate_erf(&pred, &obj.position(), self.config.accuracy);

                    if self.config.disallow_metaphase_anaphase_linking
                        && last_label == ObjectStateLabel::Metaphase as u32
                        && obj.label == ObjectStateLabel::Anaphase as u32
                    {
                        prob_assign = 0.0;
                    }

                    let prior_assign = col_prob[(*idx, 0)];
                    let (safe_update, posterior) = safe_bayesian_update(
                        prior_assign,
                        prob_assign,
                        self.config.prob_not_assign,
                    );

                    for (_, other_idx) in &neighbors {
                        if *other_idx != *idx {
                            col_prob[(*other_idx, 0)] *= safe_update;
                        }
                    }
                    col_prob[(n_obs, 0)] *= safe_update;
                    col_prob[(*idx, 0)] = posterior;
                }
            }

            for row in 0..=n_obs {
                belief[(row, col)] = col_prob[(row, 0)];
            }
        }
    }

    /// Associates tracks with observations based on belief scores and distance thresholds.
    fn link(&mut self, belief: &DMatrix<f64>, new_objects: &[TrackObject]) {
        let n_obs = new_objects.len();
        let mut unassigned_tracks: HashSet<usize> = self.active_indices.iter().copied().collect();

        // candidate_map: object_idx -> Vec<(trk_idx, prob)>
        let mut candidate_map: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();

        for (col, &trk_idx) in self.active_indices.iter().enumerate() {
            let mut best_row = n_obs;
            let mut max_prob = -1.0;

            for row in 0..=n_obs {
                let p = belief[(row, col)];
                if p > max_prob {
                    max_prob = p;
                    best_row = row;
                }
            }

            if best_row < n_obs {
                candidate_map.entry(best_row).or_default().push((trk_idx, max_prob));
            } else {
                // Lost hypothesis won
                self.manager.tracks[trk_idx].append_dummy(self.config.use_motion_features);
                unassigned_tracks.remove(&trk_idx);
                self.statistics.n_lost += 1;
            }
        }

        // Resolve correspondences per observation
        for (obj_idx, obj) in new_objects.iter().enumerate() {
            let links = candidate_map.get(&obj_idx);

            match links {
                None => {
                    // No tracks proposed this object -> start a new tracklet!
                    let new_trk = Tracklet::new(
                        self.next_track_id(),
                        obj.clone(),
                        self.config.max_lost,
                        self.motion_model.clone(),
                    );
                    self.manager.push_track(new_trk);
                }
                Some(candidates) => {
                    // Resolve conflict if multiple tracks linked to this object
                    let mut best_trk_idx = None;
                    let mut best_prob = -1.0;

                    for &(t_idx, prob) in candidates {
                        if prob > best_prob && unassigned_tracks.contains(&t_idx) {
                            best_prob = prob;
                            best_trk_idx = Some(t_idx);
                        }
                    }

                    if let Some(trk_idx) = best_trk_idx {
                        let trk = &self.manager.tracks[trk_idx];
                        let dist = (trk.position() - obj.position()).norm();

                        if dist <= self.config.max_search_radius {
                            self.manager.tracks[trk_idx].append(obj.clone(), self.config.use_motion_features);
                            unassigned_tracks.remove(&trk_idx);
                        }
                    } else {
                        // All candidates were already assigned elsewhere -> start new tracklet
                        let new_trk = Tracklet::new(
                            self.next_track_id(),
                            obj.clone(),
                            self.config.max_lost,
                            self.motion_model.clone(),
                        );
                        self.manager.push_track(new_trk);
                    }
                }
            }
        }

        // Remaining active tracks that weren't assigned get dummy points
        for trk_idx in unassigned_tracks {
            self.manager.tracks[trk_idx].append_dummy(self.config.use_motion_features);
        }
    }

    /// Runs tracking to completion across all ingested frames.
    pub fn track_all(&mut self) -> Result<TrackingInfo, TrackingError> {
        if !self.initialised {
            self.initialise()?;
        }
        let total_frames = self.frames.len();
        self.step(total_frames + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn test_motion_model() -> MotionModel {
        // 6 states: x, y, z, vx, vy, vz; 3 measurements: x, y, z
        let mut a = DMatrix::identity(6, 6);
        a[(0, 3)] = 1.0;
        a[(1, 4)] = 1.0;
        a[(2, 5)] = 1.0;

        let mut h = DMatrix::zeros(3, 6);
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;
        h[(2, 2)] = 1.0;

        let p = DMatrix::identity(6, 6) * 10.0;
        let r = DMatrix::identity(3, 3) * 0.1;
        let q = DMatrix::identity(6, 6) * 0.1;

        MotionModel::new(a, h, p, r, q)
    }

    #[test]
    fn test_simple_tracker_synthetic() {
        let mm = test_motion_model();
        let mut tracker = BayesianTracker::new(mm, TrackerConfig::default());

        // Create linear trajectory of 5 observations across 5 frames
        let mut objects = Vec::new();
        for t in 0..5 {
            let obj = TrackObject::new(t as i64, 10.0 + (t as f64) * 2.0, 10.0, 0.0, t);
            objects.push(obj);
        }

        tracker.append(objects);
        let info = tracker.track_all().expect("Tracking should succeed");

        assert_eq!(info.n_tracks, 1);
        assert_eq!(tracker.manager.tracks.len(), 1);
        assert_eq!(tracker.manager.tracks[0].length(), 5);
        assert_eq!(tracker.manager.tracks[0].track.last().unwrap().t, 4);
    }

    #[test]
    fn test_tracker_with_missing_frame_dummy_insertion() {
        let mm = test_motion_model();
        let mut tracker = BayesianTracker::new(mm, TrackerConfig::default());

        // Trajectory with missing frame at t = 2 (frames: 0, 1, 3)
        let mut objects = Vec::new();
        objects.push(TrackObject::new(10, 10.0, 10.0, 0.0, 0));
        objects.push(TrackObject::new(11, 12.0, 10.0, 0.0, 1));
        objects.push(TrackObject::new(12, 16.0, 10.0, 0.0, 3));

        tracker.append(objects);
        let info = tracker.track_all().expect("Tracking should succeed");

        assert_eq!(info.n_tracks, 1);
        let trk = &tracker.manager.tracks[0];
        // Length should be 4: frame 0, frame 1, dummy in frame 2, frame 3
        assert_eq!(trk.length(), 4);
        assert!(trk.track[2].dummy);
        assert_eq!(trk.track[2].t, 2);
    }
}
