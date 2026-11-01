//! Mixed-Integer Linear Programming (MILP) track association optimizer.

use std::collections::HashSet;
use minilp::{ComparisonOp, OptimizationDirection, Problem};

use crate::hypothesis::Hypothesis;
use crate::types::HypothesisType;

/// TrackOptimiser solves the global data association problem using binary integer linear programming.
pub struct TrackOptimiser<'a> {
    pub hypotheses: &'a [Hypothesis],
}

impl<'a> TrackOptimiser<'a> {
    pub fn new(hypotheses: &'a [Hypothesis]) -> Self {
        Self { hypotheses }
    }

    /// Solves the global association problem and returns the indices of selected hypotheses.
    pub fn optimise(&self) -> Vec<usize> {
        if self.hypotheses.is_empty() {
            return Vec::new();
        }

        let n_hypotheses = self.hypotheses.len();
        let max_id = self.hypotheses.iter().map(|h| h.id).max().unwrap_or(0);
        if max_id == 0 {
            return Vec::new();
        }
        let n_tracks = max_id;

        // Build constraint rows: row_constraints[r] = Vec<(hypothesis_idx, 1.0)>
        // r in 0..n_tracks: Outflow constraints (termination/link/branch)
        // r in n_tracks..2*n_tracks: Inflow constraints (initialization/link/branch-children)
        let mut row_constraints: Vec<Vec<usize>> = vec![Vec::new(); 2 * n_tracks];

        for (k, h) in self.hypotheses.iter().enumerate() {
            let trk = h.id.saturating_sub(1);
            if trk >= n_tracks {
                continue;
            }

            match h.hypothesis_type {
                HypothesisType::FalsePositive => {
                    row_constraints[trk].push(k);
                    row_constraints[n_tracks + trk].push(k);
                }
                HypothesisType::Init
                | HypothesisType::InitBorder
                | HypothesisType::InitFront
                | HypothesisType::InitLazy => {
                    row_constraints[n_tracks + trk].push(k);
                }
                HypothesisType::Term
                | HypothesisType::TermBorder
                | HypothesisType::TermBack
                | HypothesisType::TermLazy
                | HypothesisType::Apop => {
                    row_constraints[trk].push(k);
                }
                HypothesisType::Link => {
                    let trk_j = h.link_id.saturating_sub(1);
                    row_constraints[trk].push(k);
                    if trk_j < n_tracks {
                        row_constraints[n_tracks + trk_j].push(k);
                    }
                }
                HypothesisType::Branch => {
                    let c1 = h.child_one_id.saturating_sub(1);
                    let c2 = h.child_two_id.saturating_sub(1);
                    row_constraints[trk].push(k);
                    if c1 < n_tracks {
                        row_constraints[n_tracks + c1].push(k);
                    }
                    if c2 < n_tracks {
                        row_constraints[n_tracks + c2].push(k);
                    }
                }
                HypothesisType::Merge => {
                    let p1 = h.parent_one_id.saturating_sub(1);
                    let p2 = h.parent_two_id.saturating_sub(1);
                    row_constraints[n_tracks + trk].push(k);
                    if p1 < n_tracks {
                        row_constraints[p1].push(k);
                    }
                    if p2 < n_tracks {
                        row_constraints[p2].push(k);
                    }
                }
                _ => {}
            }
        }

        // Branch and bound search for binary optimal solution
        self.branch_and_bound(n_tracks, n_hypotheses, &row_constraints)
    }

    fn solve_lp_subproblem(
        &self,
        n_tracks: usize,
        n_hypotheses: usize,
        row_constraints: &[Vec<usize>],
        fixed_zero: &HashSet<usize>,
        fixed_one: &HashSet<usize>,
    ) -> Option<(f64, Vec<f64>)> {
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let mut vars = Vec::with_capacity(n_hypotheses);

        for k in 0..n_hypotheses {
            let bounds = if fixed_zero.contains(&k) {
                (0.0, 0.0)
            } else if fixed_one.contains(&k) {
                (1.0, 1.0)
            } else {
                (0.0, 1.0)
            };

            let v = problem.add_var(self.hypotheses[k].probability, bounds);
            vars.push(v);
        }

        for r in 0..(2 * n_tracks) {
            let hyp_indices = &row_constraints[r];
            if hyp_indices.is_empty() {
                continue;
            }

            let terms: Vec<(minilp::Variable, f64)> =
                hyp_indices.iter().map(|&idx| (vars[idx], 1.0)).collect();

            problem.add_constraint(&terms, ComparisonOp::Eq, 1.0);
        }

        match problem.solve() {
            Ok(solution) => {
                let obj = solution.objective();
                let values: Vec<f64> = vars.iter().map(|&v| solution[v]).collect();
                Some((obj, values))
            }
            Err(_) => None,
        }
    }

    fn branch_and_bound(
        &self,
        n_tracks: usize,
        n_hypotheses: usize,
        row_constraints: &[Vec<usize>],
    ) -> Vec<usize> {
        let mut stack = Vec::new();
        stack.push((HashSet::new(), HashSet::new()));

        let mut best_obj = f64::NEG_INFINITY;
        let mut best_solution = Vec::new();

        while let Some((fixed_zero, fixed_one)) = stack.pop() {
            if let Some((obj, values)) = self.solve_lp_subproblem(
                n_tracks,
                n_hypotheses,
                row_constraints,
                &fixed_zero,
                &fixed_one,
            ) {
                if obj <= best_obj {
                    continue; // Prune branch: upper bound cannot exceed best known integer solution
                }

                // Check if all variables are integers
                let mut most_fractional_idx = None;
                let mut max_frac_dist = 0.0;

                for (idx, &val) in values.iter().enumerate() {
                    let dist_to_int = (val - val.round()).abs();
                    if dist_to_int > 1e-4 {
                        let frac_score = 0.5 - (val - 0.5).abs();
                        if frac_score > max_frac_dist {
                            max_frac_dist = frac_score;
                            most_fractional_idx = Some(idx);
                        }
                    }
                }

                match most_fractional_idx {
                    None => {
                        // Integer solution found!
                        if obj > best_obj {
                            best_obj = obj;
                            best_solution = values
                                .iter()
                                .enumerate()
                                .filter(|(_, &v)| v > 0.5)
                                .map(|(i, _)| i)
                                .collect();
                        }
                    }
                    Some(frac_idx) => {
                        // Branch on the most fractional variable
                        let mut branch_zero_0 = fixed_zero.clone();
                        branch_zero_0.insert(frac_idx);
                        stack.push((branch_zero_0, fixed_one.clone()));

                        let mut branch_one_1 = fixed_one.clone();
                        branch_one_1.insert(frac_idx);
                        stack.push((fixed_zero, branch_one_1));
                    }
                }
            }
        }

        best_solution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimiser_simple_link() {
        // Track 1 and Track 2
        // H0: T1 FP (prob -10.0)
        // H1: T1 Init (prob -1.0)
        // H2: T1 Term (prob -1.0)
        // H3: T2 FP (prob -10.0)
        // H4: T2 Init (prob -1.0)
        // H5: T2 Term (prob -1.0)
        // H6: T1 -> T2 Link (prob -0.1)

        let mut hypotheses = Vec::new();
        hypotheses.push(Hypothesis::new(HypothesisType::FalsePositive, 1, -10.0));
        hypotheses.push(Hypothesis::new(HypothesisType::Init, 1, -1.0));
        hypotheses.push(Hypothesis::new(HypothesisType::Term, 1, -1.0));

        hypotheses.push(Hypothesis::new(HypothesisType::FalsePositive, 2, -10.0));
        hypotheses.push(Hypothesis::new(HypothesisType::Init, 2, -1.0));
        hypotheses.push(Hypothesis::new(HypothesisType::Term, 2, -1.0));

        let mut link = Hypothesis::new(HypothesisType::Link, 1, -0.1);
        link.link_id = 2;
        hypotheses.push(link);

        let opt = TrackOptimiser::new(&hypotheses);
        let selected = opt.optimise();

        // Optimal association:
        // T1 Init (H1: -1.0)
        // T1 -> T2 Link (H6: -0.1)
        // T2 Term (H5: -1.0)
        // Total cost = -2.1
        // (Alternative without link: H1(-1) + H2(-1) + H4(-1) + H5(-1) = -4.0)

        assert!(selected.contains(&1), "Should initialize T1");
        assert!(selected.contains(&6), "Should link T1 -> T2");
        assert!(selected.contains(&5), "Should terminate T2");
        assert_eq!(selected.len(), 3);
    }
}
