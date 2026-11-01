//! Tracklet and graph manager.

use std::collections::{HashSet, VecDeque};

use crate::tracklet::Tracklet;
use crate::types::TrackObject;

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
}
