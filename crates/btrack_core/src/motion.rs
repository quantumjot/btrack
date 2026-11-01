//! Kalman filter motion model implementation.

use nalgebra::{DMatrix, DVector, Vector3};

use crate::types::{Prediction, TrackObject};

/// Implements a discrete linear Kalman filter for object motion prediction and state tracking.
#[derive(Debug, Clone)]
pub struct MotionModel {
    /// State transition matrix (states x states)
    pub a: DMatrix<f64>,
    /// Observation matrix (measurements x states)
    pub h: DMatrix<f64>,
    /// State covariance matrix (states x states)
    pub p: DMatrix<f64>,
    /// Process noise covariance (states x states)
    pub q: DMatrix<f64>,
    /// Measurement noise covariance (measurements x measurements)
    pub r: DMatrix<f64>,

    /// Identity matrix (states x states)
    identity: DMatrix<f64>,

    /// State estimates
    pub x_hat: DVector<f64>,

    /// Number of state variables and measurements
    pub states: usize,
    pub measurements: usize,

    /// 3D motion displacement vector
    pub motion_vector: Vector3<f64>,

    /// Filter initialization flag
    pub initialised: bool,
    pub dt: f64,
    pub accuracy: f64,
    pub max_lost: usize,
    pub prob_not_assign: f64,
}

impl MotionModel {
    pub fn new(
        a: DMatrix<f64>,
        h: DMatrix<f64>,
        p: DMatrix<f64>,
        r: DMatrix<f64>,
        q: DMatrix<f64>,
    ) -> Self {
        let states = a.nrows();
        let measurements = h.nrows();
        assert_eq!(a.ncols(), states);
        assert_eq!(h.ncols(), states);
        assert_eq!(p.nrows(), states);
        assert_eq!(p.ncols(), states);
        assert_eq!(q.nrows(), states);
        assert_eq!(q.ncols(), states);
        assert_eq!(r.nrows(), measurements);
        assert_eq!(r.ncols(), measurements);

        let identity = DMatrix::identity(states, states);
        let x_hat = DVector::zeros(states);

        Self {
            a,
            h,
            p,
            q,
            r,
            identity,
            x_hat,
            states,
            measurements,
            motion_vector: Vector3::zeros(),
            initialised: true,
            dt: 1.0,
            accuracy: 2.0,
            max_lost: 5,
            prob_not_assign: 0.01,
        }
    }

    /// Sets the initial state position from an observation.
    pub fn setup(&mut self, obj: &TrackObject) {
        if self.states >= 3 {
            self.x_hat[0] = obj.x;
            self.x_hat[1] = obj.y;
            self.x_hat[2] = obj.z;
        }
    }

    /// Returns the current state prediction and covariance.
    pub fn predict(&self) -> Prediction {
        Prediction::new(self.x_hat.clone(), self.p.clone())
    }

    /// Runs a Kalman filter update with a new observation (or dummy observation).
    pub fn update(&mut self, obj: &TrackObject) {
        // 1. Time update (predict step)
        let mut x_hat_new = &self.a * &self.x_hat;
        self.p = &self.a * &self.p * self.a.transpose() + &self.q;

        // If this is a dummy object (no new measurement), advance state estimate without observation update
        if obj.dummy {
            self.x_hat = x_hat_new;
            return;
        }

        // 2. Measurement update
        let h_trans = self.h.transpose();
        let s = &self.h * &self.p * &h_trans + &self.r;

        let s_inv = match s.try_inverse() {
            Some(inv) => inv,
            None => {
                // Fallback in case innovation covariance is singular: do pseudo-inverse or keep prior
                self.x_hat = x_hat_new;
                return;
            }
        };

        let k = &self.p * &h_trans * s_inv;
        let mut z = DVector::zeros(self.measurements);
        if self.measurements > 0 {
            z[0] = obj.x;
        }
        if self.measurements > 1 {
            z[1] = obj.y;
        }
        if self.measurements > 2 {
            z[2] = obj.z;
        }
        let innovation = z - &self.h * &x_hat_new;
        x_hat_new = &x_hat_new + &k * innovation;

        // Update motion vector (displacement in position)
        if self.states >= 3 {
            self.motion_vector = Vector3::new(
                x_hat_new[0] - self.x_hat[0],
                x_hat_new[1] - self.x_hat[1],
                x_hat_new[2] - self.x_hat[2],
            );
        }

        // Update error covariance P = (I - K * H) * P
        self.p = (&self.identity - &k * &self.h) * &self.p;
        self.x_hat = x_hat_new;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_filter_tracking() {
        // Simple 1D constant velocity model: state = [x, vx], measurement = [x]
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]);
        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let p = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);
        let q = DMatrix::from_row_slice(2, 2, &[0.01, 0.0, 0.0, 0.01]);

        let mut model = MotionModel::new(a, h, p, r, q);
        let obj0 = TrackObject::new(1, 0.0, 0.0, 0.0, 0);
        model.setup(&obj0);

        let obj1 = TrackObject::new(2, 1.05, 0.0, 0.0, 1);
        model.update(&obj1);

        assert!((model.x_hat[0] - 1.05).abs() < 0.2);
        assert!(model.x_hat[1] > 0.0); // positive velocity estimated
    }
}
