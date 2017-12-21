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

#include "eigen/Eigen/Dense"
#include <vector>

#include "types.hpp"
#include "hypothesis.hpp"
#include "tracklet.hpp"






// A track manager class, behaves as if a simple vector of TrackletPtrs, but
// contains functions to allow track mergin and renaming. We only need a subset
// of the vector functionality, so no need to subclass
class TrackManager
{
  public:
    // default constructors and destructors
    TrackManager() {};
    ~TrackManager() {};

    // return the number of tracks
    size_t size() const {
      return m_tracks.size();
    }

    // return a track by index
    inline TrackletPtr operator[] (const unsigned int idx) const {
      return m_tracks[idx];
    };

    // push a tracklet onto the stack
    inline void push_back(const TrackletPtr a_obj) {
      m_tracks.push_back(a_obj);
    }

    // reserve space for new tracks
    inline void reserve(const unsigned int a_reserve){
      m_tracks.reserve(a_reserve);
    }

    // test whether the track manager is empty
    inline bool empty() const {
      return m_tracks.empty();
    }

    // merges all tracks that have a link hypothesis, renumbers others and sets
    // parent and root properties
    void merge(const std::vector<Hypothesis>);

  private:
    // a vector of tracklet objects
    std::vector<TrackletPtr> m_tracks;
};

// Make a manager pointer type
typedef std::shared_ptr<TrackManager> ManagerPtr;

#endif
