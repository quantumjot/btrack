/*
--------------------------------------------------------------------------------
 Name:     btrack
 Purpose:  A multi object tracking library, specifically used to reconstruct
           tracks in crowded fields. Here we use a probabilistic network of
           information to perform the trajectory linking. This method uses
           positional and visual information for track linking.

 Authors:  Alan R. Lowe (arl) a.lowe@ucl.ac.uk

 License:  See LICENSE.md

 Created:  14/08/2014
--------------------------------------------------------------------------------
*/

#ifndef _MANAGER_H_INCLUDED_
#define _MANAGER_H_INCLUDED_

#include <stack>
#include <vector>

#include "defs.h"
#include "hypothesis.h"
#include "tracklet.h"
#include "types.h"

// make a joining hypothesis (note: LinkHypothesis is used by the tracker...)
typedef std::pair<TrackletPtr, TrackletPtr> JoinHypothesis;

// make a branching hypothesis
typedef std::tuple<TrackletPtr, TrackletPtr, TrackletPtr> BranchHypothesis;

// make a merging hypothesis
typedef std::tuple<TrackletPtr, TrackletPtr, TrackletPtr> MergeHypothesis;

// compare two hypotheses, used for sorting by start time
bool compare_hypothesis_time(const Hypothesis &h_one, const Hypothesis &h_two);

// merge two tracks
void join_tracks(const TrackletPtr &parent_trk, const TrackletPtr &join_trk);

// set a branch between the parent and children
void branch_tracks(const BranchHypothesis &branch);

// merge tracks
void merge_tracks(const MergeHypothesis &merge);

// Lineage tree node, used for building trees
class LineageTreeNode {
   public:
    LineageTreeNode(){};
    ~LineageTreeNode(){};

    LineageTreeNode(TrackletPtr a_track) { m_track = a_track; };

    bool has_children(void) const { return m_track->has_children(); };

    TrackletPtr m_track;

    // pointers to left and right nodes
    LineageTreeNode *m_left;
    LineageTreeNode *m_right;

    // store the generational depth, default depth is always zero
    unsigned int m_depth = 0;
};

// A track manager class, behaves as if a simple vector of TrackletPtrs, but
// contains functions to allow track merging and renaming. We only need a subset
// of the vector functionality, so no need to subclass.
class TrackManager {
   public:
    // default constructors and destructors
    TrackManager() {
        m_graph_nodes.reserve(RESERVE_GRAPH_NODES);
        m_tracks.reserve(RESERVE_ALL_TRACKS);
        // m_graph_edges.reserve(RESERVE_GRAPH_EDGES);
    };
    virtual ~TrackManager(){};

    void clear(void) {
        //
    }

    inline size_t num_nodes() const { return this->m_graph_nodes.size(); }

    // return the number of tracks
    inline size_t num_tracks(void) const { return this->m_tracks.size(); }

    // return the number of graph edges
    // this includes both the greedy and ILP edges
    size_t num_edges(void) const;

    // return the number of hypotheses
    inline size_t num_hypotheses(void) const {
        return this->m_hypotheses.size();
    }

    // return a track by index
    inline TrackletPtr operator[](const size_t a_idx) const {
        return m_tracks[a_idx];
    }

    // get a node
    TrackObjectPtr get_node(const size_t a_idx) const {
        return m_graph_nodes[a_idx];
    }

    // return a track by index
    TrackletPtr get_track(const size_t a_idx) const { return m_tracks[a_idx]; }

    // return track by ID
    TrackletPtr get_track_by_ID(const size_t a_ID) const;

    // return a dummy object by index
    TrackObjectPtr get_dummy(const int a_idx) const;

    // // return a hypothesis by index
    // Hypothesis get_hypothesis(const size_t a_idx) const {
    //   return m_hypotheses[a_idx];
    // }

    // push a new node
    inline void push_node(const TrackObjectPtr &a_node) {
        m_graph_nodes.push_back(a_node);
    }

    // push a new track
    inline void push_track(const TrackletPtr &a_trk) {
        m_tracks.push_back(a_trk);
    }

    // push a new hypothesis
    inline void push_hypothesis(const Hypothesis &a_hypotheses) {
        m_hypotheses.push_back(a_hypotheses);
    }

    // // reserve space for new tracks
    // inline void reserve(const unsigned int a_reserve) {
    //   m_tracks.reserve(a_reserve);
    // }

    // // test whether thse track manager is empty
    // inline bool empty() const {
    //   return m_tracks.empty();
    // }

    // set a flag to store the candidate graph
    void set_store_candidate_graph(const bool a_store_graph);

    // return the flag to store the candidate graph
    inline bool get_store_candidate_graph(void) const {
        return m_store_candidate_graph;
    }

    // add a graph edge
    void push_edge(const TrackObjectPtr &a_node_src,
                   const TrackObjectPtr &a_node_dst, const float a_score,
                   const unsigned int a_edge_type);

    // get a graph edge from the vector
    PyGraphEdge get_edge(const size_t idx) const;

    // sort nodes
    void sort_nodes(void) {
        // sort the objects vector by time
        std::sort(m_graph_nodes.begin(), m_graph_nodes.end(), compare_obj_time);
    }

    // build trees from the data
    void build_trees();

    // finalise the track output, giving dummy objects their unique (orthogonal)
    // IDs for later retrieval, and any other cleanup required.
    void finalise();

    // merges all tracks that have a link hypothesis, renumbers others and sets
    // parent and root properties
    void merge(const std::vector<Hypothesis> &a_hypotheses);

    // split a track with certain transitions
    void split(const TrackletPtr &a_trk, const unsigned int a_label_i,
               const unsigned int a_label_j);

   private:
    // // track maintenance
    // void renumber();
    // void purge();

    // a vector of track objects (i.e. nodes)
    std::vector<TrackObjectPtr> m_graph_nodes;

    // a vector of tracklet objects
    std::vector<TrackletPtr> m_tracks;

    // set up some space for the trees, should we want these later
    std::vector<LineageTreeNode> m_trees;

    // a vector of dummy objects
    std::vector<TrackObjectPtr> m_dummies;

    // store the graph edges, i.e. intermediate output of Bayesian updates
    // and the graph hypotheses
    std::vector<PyGraphEdge> m_graph_edges;

    // store the raw hypotheses from the hypothesis engine
    std::vector<Hypothesis> m_hypotheses;

    // make hypothesis maps
    HypothesisMap<JoinHypothesis> m_links;
    HypothesisMap<BranchHypothesis> m_branches;

    bool m_store_candidate_graph = false;
};

#endif
