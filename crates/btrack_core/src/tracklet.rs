//! Tracklet container representing a single contiguous object trajectory.

use nalgebra::Vector3;

use crate::motion::MotionModel;
use crate::types::{HypothesisType, Prediction, TrackObject, MAX_LOST};

/// A tracklet container managing observation history, motion model predictions, and lineage IDs.
#[derive(Debug, Clone)]
pub struct Tracklet {
    pub id: usize,
    pub track: Vec<TrackObject>,
    pub motion_model: MotionModel,
    pub kalman: Vec<Prediction>,
    pub prediction: Vec<Prediction>,
    pub lost: usize,
    pub max_lost: usize,
    pub root: usize,
    pub parent: usize,
    pub child_one: usize,
    pub child_two: usize,
    pub renamed_id: usize,
    pub fate: HypothesisType,
    pub generation: usize,
    pub remove_flag: bool,
}

impl Tracklet {
    pub fn new(id: usize, initial_object: TrackObject, max_lost: usize, mut motion_model: MotionModel) -> Self {
        motion_model.setup(&initial_object);

        let mut trk = Self {
            id,
            track: Vec::new(),
            motion_model,
            kalman: Vec::new(),
            prediction: Vec::new(),
            lost: 0,
            max_lost: if max_lost == 0 { MAX_LOST } else { max_lost },
            root: 0,
            parent: 0,
            child_one: 0,
            child_two: 0,
            renamed_id: id,
            fate: HypothesisType::Undefined,
            generation: 0,
            remove_flag: false,
        };

        trk.append(initial_object, false);
        trk
    }

    /// Appends a new object to the trajectory, optionally updating the motion model.
    pub fn append(&mut self, obj: TrackObject, update_model: bool) {
        let is_dummy = obj.dummy;
        self.track.push(obj);

        if update_model {
            self.motion_model.update(self.track.last().unwrap());
        }

        // Store prediction before update and Kalman state
        self.prediction.push(self.predict());
        self.kalman.push(self.motion_model.predict());

        if is_dummy {
            self.lost += 1;
        } else {
            self.lost = 0;
        }
    }

    /// Appends a dummy object to fill a missing observation.
    pub fn append_dummy(&mut self, update_position: bool) {
        if self.lost > self.max_lost {
            return;
        }

        let last = self.track.last().expect("Tracklet cannot be empty");
        let (x, y, z) = if update_position {
            let p = self.predict();
            (p.mu[0], p.mu[1], p.mu[2])
        } else {
            (last.x, last.y, last.z)
        };

        let mut dummy = TrackObject::new_dummy(last.t + 1, x, y, z);
        dummy.label = last.label;
        dummy.features = last.features.clone();

        self.append(dummy, true);
    }

    /// Trims any trailing dummy objects from the end of the tracklet.
    pub fn trim(&mut self) -> bool {
        let mut trimmed = false;
        while let Some(last) = self.track.last() {
            if last.dummy {
                self.track.pop();
                self.prediction.pop();
                self.kalman.pop();
                trimmed = true;
            } else {
                break;
            }
        }
        trimmed
    }

    /// Current predicted position and covariance for future linkages.
    pub fn predict(&self) -> Prediction {
        let p = self.motion_model.predict();
        let mu_vec3 = self.position() + self.motion_model.motion_vector;
        let mu = nalgebra::DVector::from_row_slice(mu_vec3.as_slice());
        let covar = p.covar.view((0, 0), (3, 3)).into_owned();
        Prediction::new(mu, covar)
    }

    #[inline]
    pub fn length(&self) -> usize {
        self.track.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.track.is_empty()
    }

    #[inline]
    pub fn duration(&self) -> usize {
        if self.track.is_empty() {
            0
        } else {
            self.track.last().unwrap().t - self.track.first().unwrap().t
        }
    }

    #[inline]
    pub fn active(&self) -> bool {
        self.lost <= self.max_lost
    }

    #[inline]
    pub fn set_lost(&mut self) {
        self.lost = self.max_lost + 1;
    }

    #[inline]
    pub fn position(&self) -> Vector3<f64> {
        self.track.last().expect("Tracklet cannot be empty").position()
    }

    #[inline]
    pub fn has_children(&self) -> bool {
        self.child_one != 0 && self.child_two != 0 && self.child_one != self.child_two
    }
}
