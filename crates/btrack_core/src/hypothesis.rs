//! Hypothesis generation engine for global track optimization.

use crate::hyperbin::HypercubeBin;
use crate::manager::TrackManager;
use crate::tracklet::Tracklet;
use crate::types::{
    HypothesisType, ImagingVolume, ObjectStateLabel, DEFAULT_LOW_PROBABILITY,
};

/// Parameters for hypothesis probability evaluation and candidate filtering.
#[derive(Debug, Clone)]
pub struct HypothesisParams {
    pub lambda_time: f64,
    pub lambda_dist: f64,
    pub lambda_link: f64,
    pub lambda_branch: f64,
    pub eta: f64,
    pub theta_dist: f64,
    pub theta_time: f64,
    pub dist_thresh: f64,
    pub time_thresh: f64,
    pub apop_thresh: usize,
    pub segmentation_miss_rate: f64,
    pub apoptosis_rate: f64,
    pub relax: bool,
    pub hypotheses_to_generate: u32,
}

impl Default for HypothesisParams {
    fn default() -> Self {
        Self {
            lambda_time: 5.0,
            lambda_dist: 3.0,
            lambda_link: 10.0,
            lambda_branch: 50.0,
            eta: 1e-5,
            theta_dist: 20.0,
            theta_time: 5.0,
            dist_thresh: 40.0,
            time_thresh: 5.0,
            apop_thresh: 5,
            segmentation_miss_rate: 0.1,
            apoptosis_rate: 0.001,
            relax: true,
            hypotheses_to_generate: 0b1111111, // all allowed
        }
    }
}

/// Represents a single candidate association hypothesis.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    pub hypothesis_type: HypothesisType,
    pub id: usize,
    pub probability: f64, // log-likelihood
    pub link_id: usize,
    pub child_one_id: usize,
    pub child_two_id: usize,
    pub parent_one_id: usize,
    pub parent_two_id: usize,
}

impl Hypothesis {
    pub fn new(hypothesis_type: HypothesisType, id: usize, probability: f64) -> Self {
        Self {
            hypothesis_type,
            id,
            probability,
            link_id: 0,
            child_one_id: 0,
            child_two_id: 0,
            parent_one_id: 0,
            parent_two_id: 0,
        }
    }
}

#[inline]
pub fn safe_log(value: f64) -> f64 {
    if value <= 0.0 {
        DEFAULT_LOW_PROBABILITY.ln()
    } else {
        value.ln()
    }
}

/// HypothesisEngine generates track hypotheses (false positives, linkages, divisions, terminations, etc.).
pub struct HypothesisEngine<'a> {
    pub params: HypothesisParams,
    pub volume: &'a ImagingVolume,
    pub frame_range: (usize, usize),
    cube: HypercubeBin<usize>, // maps coordinates -> track index in manager
}

impl<'a> HypothesisEngine<'a> {
    pub fn new(params: HypothesisParams, volume: &'a ImagingVolume, frame_range: (usize, usize)) -> Self {
        let cube = HypercubeBin::new(params.dist_thresh, params.time_thresh);
        Self {
            params,
            volume,
            frame_range,
            cube,
        }
    }

    #[inline]
    fn hypothesis_allowed(&self, h_type: HypothesisType) -> bool {
        let bit = 1 << (h_type as u32);
        (self.params.hypotheses_to_generate & bit) != 0
    }

    fn dist_from_border(&self, trk: &Tracklet, at_start: bool) -> f64 {
        let pos = if at_start {
            trk.track.first().unwrap().position()
        } else {
            trk.track.last().unwrap().position()
        };
        self.volume.dist_from_border(&pos)
    }

    fn count_apoptosis(&self, trk: &Tracklet) -> usize {
        let mut count = 0;
        for obj in trk.track.iter().rev() {
            if obj.label == ObjectStateLabel::Apoptosis as u32 {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    fn p_fp(&self, trk: &Tracklet) -> f64 {
        let len = 1 + trk.duration();
        self.params.segmentation_miss_rate.powi(len as i32)
    }

    fn p_tp(&self, trk: &Tracklet) -> f64 {
        1.0 - self.p_fp(trk)
    }

    fn p_init_border(&self, trk: &Tracklet) -> f64 {
        let mut dist = self.dist_from_border(trk, true);
        dist = dist.min(self.params.theta_dist);

        if dist < self.params.theta_dist || self.params.relax {
            (-dist / self.params.lambda_dist).exp()
        } else {
            self.params.eta
        }
    }

    fn p_init_front(&self, trk: &Tracklet) -> f64 {
        let t = trk.track.first().unwrap().t;
        if t < self.frame_range.0 + self.params.theta_time as usize {
            let dt = (t - self.frame_range.0 + 1) as f64;
            (-dt / self.params.lambda_time).exp()
        } else {
            self.params.eta
        }
    }

    fn p_term_border(&self, trk: &Tracklet) -> f64 {
        let mut dist = self.dist_from_border(trk, false);
        dist = dist.min(self.params.theta_dist);

        if dist < self.params.theta_dist || self.params.relax {
            (-dist / self.params.lambda_dist).exp()
        } else {
            self.params.eta
        }
    }

    fn p_term_back(&self, trk: &Tracklet) -> f64 {
        let t = trk.track.last().unwrap().t;
        if self.frame_range.1 >= t && (self.frame_range.1 - t) < self.params.theta_time as usize {
            let dt = (self.frame_range.1 - t) as f64;
            (-dt / self.params.lambda_time).exp()
        } else {
            self.params.eta
        }
    }

    fn p_dead(&self, _trk: &Tracklet, n_apoptosis: usize) -> f64 {
        if n_apoptosis == 0 {
            return 0.0;
        }
        1.0 - self.params.apoptosis_rate.powi(n_apoptosis as i32)
    }

    fn p_link(&self, trk1: &Tracklet, trk2: &Tracklet, d: f64) -> f64 {
        let mut penalty = 1.0;
        let last_label = trk1.track.last().unwrap().label;
        let first_label = trk2.track.first().unwrap().label;

        if (last_label == ObjectStateLabel::Metaphase as u32
            || last_label == ObjectStateLabel::Prometaphase as u32)
            && first_label == ObjectStateLabel::Anaphase as u32
        {
            penalty = 10.0;
        }

        (-d * penalty / self.params.lambda_link).exp()
    }

    fn p_branch(&self, trk: &Tracklet, c0: &Tracklet, c1: &Tracklet) -> f64 {
        let d_c0 = c0.track.first().unwrap().position() - trk.track.last().unwrap().position();
        let d_c1 = c1.track.first().unwrap().position() - trk.track.last().unwrap().position();

        let norm_c0 = d_c0.norm();
        let norm_c1 = d_c1.norm();
        let dot_product = if norm_c0 > 1e-9 && norm_c1 > 1e-9 {
            (d_c0 / norm_c0).dot(&(d_c1 / norm_c1))
        } else {
            0.0
        };

        let parent_label = trk.track.last().unwrap().label;
        let c0_label = c0.track.first().unwrap().label;
        let c1_label = c1.track.first().unwrap().label;

        let weight = if parent_label == ObjectStateLabel::Metaphase as u32 {
            if c0_label == ObjectStateLabel::Anaphase as u32 && c1_label == ObjectStateLabel::Anaphase as u32 {
                0.01 // Best division match
            } else if c0_label == ObjectStateLabel::Anaphase as u32 || c1_label == ObjectStateLabel::Anaphase as u32 {
                0.1
            } else {
                2.0
            }
        } else if c0_label == ObjectStateLabel::Anaphase as u32 && c1_label == ObjectStateLabel::Anaphase as u32 {
            1.0
        } else if c0_label == ObjectStateLabel::Anaphase as u32 || c1_label == ObjectStateLabel::Anaphase as u32 {
            2.0
        } else {
            let n_apop_c0 = self.count_apoptosis(c0);
            let n_apop_c1 = self.count_apoptosis(c1);
            5.0 + 10.0 * self.p_dead(c0, n_apop_c0) + 10.0 * self.p_dead(c1, n_apop_c1)
        };

        let daughter_angle = 1.0 - ((1.0 - statrs::function::erf::erf(dot_product * 3.0)) / 2.0);
        (-weight * daughter_angle / (2.0 * self.params.lambda_branch)).exp()
    }

    /// Evaluates and generates all association hypotheses for tracks in the manager.
    pub fn create(&mut self, manager: &TrackManager) -> Vec<Hypothesis> {
        let mut hypotheses = Vec::with_capacity(manager.num_tracks() * 5);
        if manager.num_tracks() == 0 {
            return hypotheses;
        }

        self.cube.clear();
        for (idx, trk) in manager.tracks.iter().enumerate() {
            if let Some(first) = trk.track.first() {
                self.cube.add(&first.position(), first.t, idx);
            }
        }

        for (i, trk) in manager.tracks.iter().enumerate() {
            // 1. False Positive
            let log_p_fp = safe_log(self.p_fp(trk));
            hypotheses.push(Hypothesis::new(HypothesisType::FalsePositive, trk.id, log_p_fp));

            let log_p_tp_half = 0.5 * safe_log(self.p_tp(trk));

            // 2. Initialization
            if self.hypothesis_allowed(HypothesisType::Init) {
                let d_start = self.dist_from_border(trk, true);
                let first_t = trk.track.first().unwrap().t;

                if self.params.relax
                    || first_t < self.frame_range.0 + self.params.theta_time as usize
                    || d_start < self.params.theta_dist
                {
                    let p_border = self.p_init_border(trk);
                    let p_front = self.p_init_front(trk);

                    if p_border > p_front {
                        let h_type = if d_start >= self.params.theta_dist {
                            HypothesisType::InitLazy
                        } else {
                            HypothesisType::InitBorder
                        };
                        hypotheses.push(Hypothesis::new(h_type, trk.id, safe_log(p_border) + log_p_tp_half));
                    } else {
                        let h_type = if first_t >= self.frame_range.0 + self.params.theta_time as usize {
                            HypothesisType::InitLazy
                        } else {
                            HypothesisType::InitFront
                        };
                        hypotheses.push(Hypothesis::new(h_type, trk.id, safe_log(p_front) + log_p_tp_half));
                    }
                }
            }

            // 3. Termination
            if self.hypothesis_allowed(HypothesisType::Term) {
                let d_stop = self.dist_from_border(trk, false);
                let last_t = trk.track.last().unwrap().t;

                if self.params.relax
                    || last_t > self.frame_range.1 - self.params.theta_time as usize
                    || d_stop < self.params.theta_dist
                {
                    let p_border = self.p_term_border(trk);
                    let p_back = self.p_term_back(trk);

                    if p_border > p_back {
                        let h_type = if d_stop >= self.params.theta_dist {
                            HypothesisType::TermLazy
                        } else {
                            HypothesisType::TermBorder
                        };
                        hypotheses.push(Hypothesis::new(h_type, trk.id, safe_log(p_border) + log_p_tp_half));
                    } else {
                        let h_type = if last_t <= self.frame_range.1 - self.params.theta_time as usize {
                            HypothesisType::TermLazy
                        } else {
                            HypothesisType::TermBack
                        };
                        hypotheses.push(Hypothesis::new(h_type, trk.id, safe_log(p_back) + log_p_tp_half));
                    }
                }
            }

            // 4. Apoptosis / Dead
            let n_apoptosis = self.count_apoptosis(trk);
            if self.hypothesis_allowed(HypothesisType::Apop) && n_apoptosis >= self.params.apop_thresh {
                let p_dead = self.p_dead(trk, n_apoptosis);
                hypotheses.push(Hypothesis::new(HypothesisType::Apop, trk.id, safe_log(p_dead) + log_p_tp_half));
            }

            // 5. Linkage and Division candidates
            let end_pos = trk.track.last().unwrap().position();
            let end_t = trk.track.last().unwrap().t;

            let candidate_indices = self.cube.get_candidates(&end_pos, end_t);
            let mut valid_conflicts = Vec::new();

            for &cand_idx in &candidate_indices {
                if cand_idx == i {
                    continue;
                }
                let cand_trk = &manager.tracks[cand_idx];
                let start_pos = cand_trk.track.first().unwrap().position();
                let start_t = cand_trk.track.first().unwrap().t;

                if start_t <= end_t {
                    continue;
                }

                let dt = (start_t - end_t) as f64;
                let d = (start_pos - end_pos).norm();

                if d > self.params.dist_thresh || dt > self.params.time_thresh {
                    continue;
                }

                if self.hypothesis_allowed(HypothesisType::Link) {
                    let p_lnk = self.p_link(trk, cand_trk, d);
                    let score = safe_log(p_lnk) + log_p_tp_half + 0.5 * safe_log(self.p_tp(cand_trk));
                    let mut h = Hypothesis::new(HypothesisType::Link, trk.id, score);
                    h.link_id = cand_trk.id;
                    hypotheses.push(h);
                }

                valid_conflicts.push(cand_idx);
            }

            // 6. Branch / Division
            if self.hypothesis_allowed(HypothesisType::Branch) && valid_conflicts.len() >= 2 {
                for p in 0..(valid_conflicts.len() - 1) {
                    let c0 = &manager.tracks[valid_conflicts[p]];
                    for q in (p + 1)..valid_conflicts.len() {
                        let c1 = &manager.tracks[valid_conflicts[q]];
                        let p_br = self.p_branch(trk, c0, c1);
                        let score = safe_log(p_br)
                            + log_p_tp_half
                            + 0.5 * safe_log(self.p_tp(c0))
                            + 0.5 * safe_log(self.p_tp(c1));

                        let mut h = Hypothesis::new(HypothesisType::Branch, trk.id, score);
                        h.child_one_id = c0.id;
                        h.child_two_id = c1.id;
                        hypotheses.push(h);
                    }
                }
            }
        }

        hypotheses
    }
}
