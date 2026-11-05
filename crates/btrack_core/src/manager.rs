//! Tracklet and graph manager.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::hypothesis::Hypothesis;
use crate::tracklet::Tracklet;
use crate::types::{HypothesisType, TrackObject};

/// Directed graph edge representation for candidate linkages.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub source: i64,
    pub target: i64,
    pub score: f64,
    pub edge_type: u32,
}

/// Node in lineage tree for generation depth and root calculation.
#[derive(Debug, Clone)]
struct LineageTreeNode {
    track_id: usize,
    depth: usize,
}

/// TrackManager manages the pool of raw observations, active/lost tracklets, and graph edges.
#[derive(Debug, Clone, Default)]
pub struct TrackManager {
    pub graph_nodes: Vec<TrackObject>,
    pub tracks: Vec<Tracklet>,
    pub dummies: Vec<TrackObject>,
    pub graph_edges: Vec<GraphEdge>,
    pub store_candidate_graph: bool,
}

impl TrackManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.graph_nodes.clear();
        self.tracks.clear();
        self.dummies.clear();
        self.graph_edges.clear();
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.graph_nodes.len()
    }

    #[inline]
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }

    #[inline]
    pub fn num_edges(&self) -> usize {
        self.graph_edges.len()
    }

    pub fn push_node(&mut self, obj: TrackObject) {
        self.graph_nodes.push(obj);
    }

    pub fn push_track(&mut self, trk: Tracklet) {
        self.tracks.push(trk);
    }

    pub fn push_edge(&mut self, source: i64, target: i64, score: f64, edge_type: u32) {
        if self.store_candidate_graph {
            self.graph_edges.push(GraphEdge {
                source,
                target,
                score,
                edge_type,
            });
        }
    }

    /// Sorts input graph nodes chronologically by frame number `t`.
    pub fn sort_nodes(&mut self) {
        self.graph_nodes.sort_by_key(|obj| obj.t);
    }

    pub fn get_track_by_id(&self, id: usize) -> Option<&Tracklet> {
        self.tracks.iter().find(|t| t.id == id)
    }

    pub fn get_track_by_id_mut(&mut self, id: usize) -> Option<&mut Tracklet> {
        self.tracks.iter_mut().find(|t| t.id == id)
    }

    /// Builds lineage trees, computing generational depths and assigning root track IDs.
    pub fn build_trees(&mut self) {
        let mut used = HashSet::new();

        // Sort tracks by start time
        self.tracks.sort_by_key(|t| t.track.first().map(|o| o.t).unwrap_or(0));

        let track_ids: Vec<usize> = self.tracks.iter().map(|t| t.id).collect();

        for root_id in track_ids {
            if used.contains(&root_id) {
                continue;
            }

            let mut queue = VecDeque::new();
            queue.push_back(LineageTreeNode {
                track_id: root_id,
                depth: 0,
            });
            used.insert(root_id);

            while let Some(node) = queue.pop_front() {
                let (child_one, child_two) = {
                    if let Some(t) = self.get_track_by_id(node.track_id) {
                        if t.has_children() {
                            (t.child_one, t.child_two)
                        } else {
                            (0, 0)
                        }
                    } else {
                        (0, 0)
                    }
                };

                if child_one != 0 && child_two != 0 {
                    let child_depth = node.depth + 1;

                    if let Some(c1) = self.get_track_by_id_mut(child_one) {
                        c1.root = root_id;
                        c1.generation = child_depth;
                    }
                    if let Some(c2) = self.get_track_by_id_mut(child_two) {
                        c2.root = root_id;
                        c2.generation = child_depth;
                    }

                    queue.push_back(LineageTreeNode {
                        track_id: child_one,
                        depth: child_depth,
                    });
                    queue.push_back(LineageTreeNode {
                        track_id: child_two,
                        depth: child_depth,
                    });

                    used.insert(child_one);
                    used.insert(child_two);
                }
            }
        }
    }

    /// Finalizes all tracks by trimming trailing dummies, assigning unique negative IDs to dummy objects,
    /// and verifying root/parent references.
    pub fn finalise(&mut self) {
        let mut dummy_id: i64 = -1;
        self.dummies.clear();

        for trk in self.tracks.iter_mut() {
            trk.trim();

            for obj in trk.track.iter_mut() {
                if obj.dummy {
                    obj.id = dummy_id;
                    dummy_id -= 1;
                    self.dummies.push(obj.clone());
                }
            }

            if trk.root == 0 {
                trk.root = trk.id;
            }
            if trk.parent == 0 {
                trk.parent = trk.id;
            }
        }
    }

    /// Merges linked tracklets, assigns branch daughter relationships, removes fragments,
    /// and recalculates lineage roots and generational depths.
    pub fn merge(&mut self, selected_hypotheses: &[Hypothesis]) {
        if selected_hypotheses.is_empty() || self.tracks.is_empty() {
            return;
        }

        let mut links: HashMap<usize, usize> = HashMap::new();
        let mut branches: Vec<(usize, usize, usize)> = Vec::new();

        for h in selected_hypotheses {
            if let Some(trk) = self.get_track_by_id_mut(h.id) {
                trk.fate = h.hypothesis_type;
            }

            match h.hypothesis_type {
                HypothesisType::Link => {
                    links.insert(h.id, h.link_id);
                }
                HypothesisType::Branch => {
                    branches.push((h.id, h.child_one_id, h.child_two_id));
                }
                _ => {}
            }
        }

        // Set division branches
        for (parent_id, c1_id, c2_id) in branches {
            if let Some(c1) = self.get_track_by_id_mut(c1_id) {
                c1.parent = parent_id;
            }
            if let Some(c2) = self.get_track_by_id_mut(c2_id) {
                c2.parent = parent_id;
            }
            if let Some(parent) = self.get_track_by_id_mut(parent_id) {
                parent.child_one = c1_id;
                parent.child_two = c2_id;
                parent.fate = HypothesisType::Branch;
            }
        }

        // Process linkages
        let mut used = HashSet::new();
        let track_ids: Vec<usize> = self.tracks.iter().map(|t| t.id).collect();

        for parent_id in track_ids {
            if !links.contains_key(&parent_id) || used.contains(&parent_id) {
                continue;
            }

            used.insert(parent_id);
            let mut curr = parent_id;

            while let Some(&next_id) = links.get(&curr) {
                used.insert(next_id);

                let (next_objects, next_kalman, next_pred, next_fate, next_c1, next_c2) = {
                    if let Some(next_trk) = self.get_track_by_id_mut(next_id) {
                        next_trk.remove_flag = true;
                        next_trk.renamed_id = parent_id;
                        (
                            std::mem::take(&mut next_trk.track),
                            std::mem::take(&mut next_trk.kalman),
                            std::mem::take(&mut next_trk.prediction),
                            next_trk.fate,
                            next_trk.child_one,
                            next_trk.child_two,
                        )
                    } else {
                        break;
                    }
                };

                if let Some(p_trk) = self.get_track_by_id_mut(parent_id) {
                    p_trk.track.extend(next_objects);
                    p_trk.kalman.extend(next_kalman);
                    p_trk.prediction.extend(next_pred);
                    p_trk.fate = next_fate;
                    if next_c1 != 0 && next_c2 != 0 {
                        p_trk.child_one = next_c1;
                        p_trk.child_two = next_c2;
                    }
                }

                curr = next_id;
            }
        }

        // Remove tracks marked for removal
        self.tracks.retain(|t| !t.remove_flag);

        // Verify reverse parent lookups
        let current_ids: Vec<usize> = self.tracks.iter().map(|t| t.id).collect();
        for id in current_ids {
            let (c1, c2) = {
                if let Some(t) = self.get_track_by_id(id) {
                    if t.has_children() {
                        (t.child_one, t.child_two)
                    } else {
                        (0, 0)
                    }
                } else {
                    (0, 0)
                }
            };

            if c1 != 0 {
                if let Some(child) = self.get_track_by_id_mut(c1) {
                    child.parent = id;
                }
            }
            if c2 != 0 {
                if let Some(child) = self.get_track_by_id_mut(c2) {
                    child.parent = id;
                }
            }
        }

        self.build_trees();
        self.finalise();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use crate::motion::MotionModel;

    fn test_motion_model() -> MotionModel {
        let a = DMatrix::identity(6, 6);
        let h = DMatrix::zeros(3, 6);
        let p = DMatrix::identity(6, 6);
        let r = DMatrix::identity(3, 3);
        let q = DMatrix::identity(6, 6);
        MotionModel::new(a, h, p, r, q)
    }

    #[test]
    fn test_manager_merge_link() {
        let mut mgr = TrackManager::new();
        let mm = test_motion_model();

        let obj1 = TrackObject::new(1, 10.0, 10.0, 0.0, 0);
        let obj2 = TrackObject::new(2, 12.0, 10.0, 0.0, 1);
        let mut t1 = Tracklet::new(1, obj1, 5, mm.clone());
        t1.append(obj2, false);

        let obj3 = TrackObject::new(3, 14.0, 10.0, 0.0, 4);
        let obj4 = TrackObject::new(4, 16.0, 10.0, 0.0, 5);
        let mut t2 = Tracklet::new(2, obj3, 5, mm.clone());
        t2.append(obj4, false);

        mgr.push_track(t1);
        mgr.push_track(t2);

        assert_eq!(mgr.num_tracks(), 2);

        let mut link_hyp = Hypothesis::new(HypothesisType::Link, 1, -0.1);
        link_hyp.link_id = 2;

        mgr.merge(&[link_hyp]);

        assert_eq!(mgr.num_tracks(), 1);
        let merged = &mgr.tracks[0];
        assert_eq!(merged.id, 1);
        assert_eq!(merged.length(), 4);
        assert_eq!(merged.track[0].id, 1);
        assert_eq!(merged.track[3].id, 4);
    }
}
