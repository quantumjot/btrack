/*
--------------------------------------------------------------------------------
 Name:     BayesianTracker
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

#include <vector>
#include <stack>

#include "types.h"
#include "hypothesis.h"
#include "tracklet.h"

#define RESERVE_ALL_TRACKS 500000

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
class LineageTreeNode
{
  public:
    LineageTreeNode() {};
    ~LineageTreeNode() {};

    LineageTreeNode(TrackletPtr a_track) {
      m_track = a_track;
    };

    bool has_children(void) const {
      return m_track->has_children();
    };

    TrackletPtr m_track;
};




// A track manager class, behaves as if a simple vector of TrackletPtrs, but
// contains functions to allow track merging and renaming. We only need a subset
// of the vector functionality, so no need to subclass.
class TrackManager
{
  public:
    // default constructors and destructors
    TrackManager() {
      m_tracks.reserve(RESERVE_ALL_TRACKS);
    };
    virtual ~TrackManager() {};

    // return the number of tracks
    size_t size() const {
      return this->m_tracks.size();
    }

    // return a track by index
    inline TrackletPtr operator[] (const unsigned int idx) const {
      return m_tracks[idx];
    };

    // return track by ID
    TrackletPtr get_track_by_ID(const unsigned int a_ID) const;

    // return a dummy object by index
    TrackObjectPtr get_dummy(const int idx) const;

    // push a tracklet onto the stack
    inline void push_back(const TrackletPtr &a_obj) {
      m_tracks.push_back(a_obj);
    }

    // reserve space for new tracks
    inline void reserve(const unsigned int a_reserve) {
      m_tracks.reserve(a_reserve);
    }

    // test whether the track manager is empty
    inline bool empty() const {
      return m_tracks.empty();
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
    void split(const TrackletPtr &a_trk,
               const unsigned int a_label_i,
               const unsigned int a_label_j);

  private:

    // track maintenance
    void renumber();
    void purge();

    // a vector of tracklet objects
    std::vector<TrackletPtr> m_tracks;

    // set upsome space for the trees, should we want these later
    std::vector<LineageTreeNode> m_trees;

    // a vector of dummy objects
    std::vector<TrackObjectPtr> m_dummies;

    // make hypothesis maps
    HypothesisMap<JoinHypothesis> m_links;
    HypothesisMap<BranchHypothesis> m_branches;
};






#endif
