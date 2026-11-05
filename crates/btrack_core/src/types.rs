//! Core data structures, types, and constants.

use nalgebra::{DMatrix, DVector, Vector3};

/// Numerical floor used to prevent zero probabilities.
pub const DEFAULT_LOW_PROBABILITY: f64 = 1e-308;
/// Default prior probability for a lost tracklet hypothesis.
pub const DEFAULT_LOST_PROBABILITY: f64 = 0.1;
/// Spatial integration accuracy bound ($\delta$) in Euclidean units.
pub const DEFAULT_ACCURACY: f64 = 2.0;
/// Probability of spatial non-assignment (clutter / false detection rate).
pub const PROB_NOT_ASSIGN: f64 = 0.01;
/// Default concentration parameter ($\kappa$) for von Mises-Fisher directional feature likelihood.
pub const DEFAULT_FEATURE_CONCENTRATION: f64 = 3.0;
/// Maximum number of consecutive frames a tracklet can remain lost before termination.
pub const MAX_LOST: usize = 5;
/// Maximum spatial search radius for candidate object linkage.
pub const MAX_SEARCH_RADIUS: f64 = 100.0;

/// Tracking errors corresponding to C++ TrackingError enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum TrackingError {
    Success = 900,
    EmptyQueue = 901,
    NoTracks = 902,
    NoUseableFrames = 903,
    TrackEmpty = 904,
    IncorrectMotionModel = 905,
    MaxLostOutOfRange = 906,
    AccuracyOutOfRange = 907,
    ProbNotAssignOutOfRange = 908,
    NotDefined = 909,
    #[default]
    None = 910,
}

/// Hypothesis types corresponding to C++ HypothesisType.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum HypothesisType {
    FalsePositive = 0,
    Init = 1,
    Term = 2,
    Link = 3,
    Branch = 4,
    Apop = 5,
    Merge = 6,
    Extrude = 7,
    InitBorder = 10,
    InitFront = 11,
    InitLazy = 12,
    TermBorder = 20,
    TermBack = 21,
    TermLazy = 22,
    Dead = 666,
    Undefined = 999,
}

impl From<u32> for HypothesisType {
    fn from(val: u32) -> Self {
        match val {
            0 => HypothesisType::FalsePositive,
            1 => HypothesisType::Init,
            2 => HypothesisType::Term,
            3 => HypothesisType::Link,
            4 => HypothesisType::Branch,
            5 => HypothesisType::Apop,
            6 => HypothesisType::Merge,
            7 => HypothesisType::Extrude,
            10 => HypothesisType::InitBorder,
            11 => HypothesisType::InitFront,
            12 => HypothesisType::InitLazy,
            20 => HypothesisType::TermBorder,
            21 => HypothesisType::TermBack,
            22 => HypothesisType::TermLazy,
            666 => HypothesisType::Dead,
            _ => HypothesisType::Undefined,
        }
    }
}

/// Object state labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ObjectStateLabel {
    Interphase = 0,
    Prometaphase = 1,
    Metaphase = 2,
    Anaphase = 3,
    Apoptosis = 4,
    Null = 5,
    Dummy = 99,
}

impl From<u32> for ObjectStateLabel {
    fn from(val: u32) -> Self {
        match val {
            0 => ObjectStateLabel::Interphase,
            1 => ObjectStateLabel::Prometaphase,
            2 => ObjectStateLabel::Metaphase,
            3 => ObjectStateLabel::Anaphase,
            4 => ObjectStateLabel::Apoptosis,
            5 => ObjectStateLabel::Null,
            _ => ObjectStateLabel::Dummy,
        }
    }
}

/// Represents a single detected object in 3D + time space.
#[derive(Debug, Clone)]
pub struct TrackObject {
    pub id: i64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub t: usize,
    pub dummy: bool,
    pub states: usize,
    pub label: u32,
    pub features: DVector<f64>,
}

impl TrackObject {
    pub fn new(id: i64, x: f64, y: f64, z: f64, t: usize) -> Self {
        Self {
            id,
            x,
            y,
            z,
            t,
            dummy: false,
            states: 0,
            label: 0,
            features: DVector::zeros(0),
        }
    }

    pub fn new_dummy(t: usize, x: f64, y: f64, z: f64) -> Self {
        Self {
            id: -1,
            x,
            y,
            z,
            t,
            dummy: true,
            states: 0,
            label: ObjectStateLabel::Dummy as u32,
            features: DVector::zeros(0),
        }
    }

    #[inline]
    pub fn position(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }
}

/// Prediction of future state from the motion model.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub mu: DVector<f64>,
    pub covar: DMatrix<f64>,
}

impl Prediction {
    pub fn new(mu: DVector<f64>, covar: DMatrix<f64>) -> Self {
        Self { mu, covar }
    }

    pub fn default_six_states() -> Self {
        Self {
            mu: DVector::zeros(6),
            covar: DMatrix::identity(6, 6),
        }
    }

    #[inline]
    pub fn position(&self) -> Vector3<f64> {
        Vector3::new(self.mu[0], self.mu[1], self.mu[2])
    }
}

/// 3D imaging bounding volume.
#[derive(Debug, Clone)]
pub struct ImagingVolume {
    pub min_xyz: Vector3<f64>,
    pub max_xyz: Vector3<f64>,
}

impl Default for ImagingVolume {
    fn default() -> Self {
        Self {
            min_xyz: Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max_xyz: Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }
}

impl ImagingVolume {
    pub fn new(x_range: (f64, f64), y_range: (f64, f64), z_range: (f64, f64)) -> Self {
        Self {
            min_xyz: Vector3::new(x_range.0, y_range.0, z_range.0),
            max_xyz: Vector3::new(x_range.1, y_range.1, z_range.1),
        }
    }

    pub fn update(&mut self, pos: &Vector3<f64>) {
        self.min_xyz[0] = self.min_xyz[0].min(pos[0]);
        self.min_xyz[1] = self.min_xyz[1].min(pos[1]);
        self.min_xyz[2] = self.min_xyz[2].min(pos[2]);

        self.max_xyz[0] = self.max_xyz[0].max(pos[0]);
        self.max_xyz[1] = self.max_xyz[1].max(pos[1]);
        self.max_xyz[2] = self.max_xyz[2].max(pos[2]);
    }

    pub fn set_volume(&mut self, bounds: &[f64; 6]) {
        self.min_xyz = Vector3::new(bounds[0], bounds[2], bounds[4]);
        self.max_xyz = Vector3::new(bounds[1], bounds[3], bounds[5]);
    }

    #[inline]
    pub fn inside(&self, pos: &Vector3<f64>) -> bool {
        pos[0] >= self.min_xyz[0]
            && pos[0] <= self.max_xyz[0]
            && pos[1] >= self.min_xyz[1]
            && pos[1] <= self.max_xyz[1]
            && pos[2] >= self.min_xyz[2]
            && pos[2] <= self.max_xyz[2]
    }

    #[inline]
    pub fn is_2d(&self) -> bool {
        (self.min_xyz[2] - self.max_xyz[2]).abs() < 1e-9
    }

    /// Calculate minimum distance from the volume border.
    pub fn dist_from_border(&self, pos: &Vector3<f64>) -> f64 {
        let mut min_dist = f64::INFINITY;
        for dim in 0..3 {
            // skip flat dimension in 2D
            if (self.min_xyz[dim] - self.max_xyz[dim]).abs() < 1e-9 {
                continue;
            }
            let d = (pos[dim] - self.min_xyz[dim]).min(self.max_xyz[dim] - pos[dim]);
            if d < min_dist {
                min_dist = d;
            }
        }
        min_dist
    }
}

/// Statistics returned from tracking execution.
#[derive(Debug, Clone, Default)]
pub struct TrackingInfo {
    pub error: TrackingError,
    pub n_tracks: usize,
    pub n_active: usize,
    pub n_conflicts: usize,
    pub n_lost: usize,
    pub t_update_belief: f32,
    pub t_update_link: f32,
    pub t_total_time: f32,
    pub p_link: f32,
    pub p_lost: f32,
    pub complete: bool,
}
