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

#ifndef _TYPES_H_INCLUDED_
#define _TYPES_H_INCLUDED_

#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "defs.h"
#include "eigen/Eigen/Dense"

// Python structure for external interface
extern "C" struct PyTrackObject {
  long ID;
  double x;
  double y;
  double z;
  unsigned int t;
  bool dummy;
  unsigned int states;
  unsigned int label;
  unsigned int n_features;
  double *features;
};

// Output back to Python with some tracking statistics
extern "C" struct PyTrackInfo {
  unsigned int error;
  unsigned int n_tracks;
  unsigned int n_active;
  unsigned int n_conflicts;
  unsigned int n_lost;
  float t_update_belief;
  float t_update_link;
  float t_total_time;
  float p_link;
  float p_lost;
  bool complete;

  // default constructor
  PyTrackInfo()
      : error(ERROR_none), n_tracks(0), n_active(0), n_conflicts(0), n_lost(0),
        t_update_belief(0), t_update_link(0), t_total_time(0), p_link(0),
        p_lost(0), complete(false){};
};

// structure to return the Bayesian belief matrix as a series of graph edges
extern "C" struct PyGraphEdge {
  long source;
  long target;
  double score;
  unsigned int type = GRAPH_EDGE_link;
};

// TrackObject stores the data of each object in the field of view. Is
// essentially a parallel of the PyTrackObject class.
class TrackObject {
public:
  // Start a new tracklet without any prior information.
  TrackObject()
      : ID(0), x(0.), y(0.), z(0.), t(0), dummy(true), label(0),
        n_features(0){};

  // Instantiate a track object from an existing PyTrackObject
  TrackObject(const PyTrackObject &trk)
      : ID(trk.ID), x(trk.x), y(trk.y), z(trk.z), t(trk.t), dummy(trk.dummy),
        label(trk.label), states(trk.states), n_features(trk.n_features),
        features(Eigen::Map<Eigen::VectorXd>(trk.features, trk.n_features)){};

  // Default destructor
  ~TrackObject(){};

  // xyzt position, dummy flag and class label, note that the ID can be
  // negative, indicating a dummy object
  long ID;
  double x;
  double y;
  double z;
  unsigned int t;
  bool dummy;
  unsigned int label;
  unsigned int states;
  unsigned int n_features;

  // store object features
  Eigen::VectorXd features;

  // return the current position of the track object
  Eigen::Vector3d position() const {
    Eigen::Vector3d p;
    p << x, y, z;
    return p;
  };

  // return this object as a pytrack object
  PyTrackObject get_pytrack_object() const {
    PyTrackObject p = PyTrackObject();
    p.ID = this->ID;
    p.x = this->x;
    p.y = this->y;
    p.z = this->z;
    p.t = this->t;
    p.dummy = this->dummy;
    p.label = this->label;
    p.states = this->states;
    return p;
  };

private:
  // Store a reference to the original object?
  // const PyTrackObject* original_object;
};

// type definition for a track object pointer, minimising copying
typedef std::shared_ptr<TrackObject> TrackObjectPtr;

// Structure for MotionModel predictions
struct Prediction {
  // position and error predictions
  Eigen::VectorXd mu;
  Eigen::MatrixXd covar;

  // constructors
  // TODO(arl): the default constructor defaults to six states
  Prediction() {
    mu.setZero(6, 1);
    covar.setIdentity(6, 6);
  }
  Prediction(const Eigen::VectorXd &a_mu, const Eigen::MatrixXd &a_covar)
      : mu(a_mu), covar(a_covar){};
};

// Comparison object to order track objects ready for tracking.
inline bool compare_obj_time(const TrackObjectPtr trackobj_1,
                             const TrackObjectPtr trackobj_2) {
  return (trackobj_1->t < trackobj_2->t);
}

// Structure to determine the imaging volume from the observations
struct ImagingVolume {
  // minimum and maximum dimensions of the volume from observations
  Eigen::Vector3d min_xyz;
  Eigen::Vector3d max_xyz;

  // default constructor
  ImagingVolume() {
    min_xyz.fill(kInfinity);
    max_xyz.fill(-kInfinity);
  };

  // update with an observation
  void update(const TrackObjectPtr obj) {
    min_xyz = min_xyz.cwiseMin(obj->position());
    max_xyz = max_xyz.cwiseMax(obj->position());
  };

  // set the imaging volume (note that this could be overwritten...)
  void set_volume(const double *a_volume) {
    assert(a_volume[0] <= a_volume[1]);
    assert(a_volume[2] <= a_volume[3]);
    assert(a_volume[4] <= a_volume[5]);

    min_xyz << a_volume[0], a_volume[2], a_volume[4];
    max_xyz << a_volume[1], a_volume[3], a_volume[5];
  }

  // test whether an object lies within the volume
  bool inside(const Eigen::Vector3d &position) const {
    if (position(0) >= min_xyz(0) && position(0) <= max_xyz(0) &&
        position(1) >= min_xyz(1) && position(1) <= max_xyz(1) &&
        position(2) >= min_xyz(2) && position(2) <= max_xyz(2)) {
      return true;
    } else {
      return false;
    }
  }

  // test whether a dataset is 2D, i.e. all z-values are the same
  bool is_2D(void) const { return (min_xyz(2) == max_xyz(2)); }
};

// Hypothesis map is a sort of dictionary of lists, and can be used to store
// multiple hypotheses related to each track. The container has a series of
// 'bins' (integer indexing) in which hypotheses can be placed (using push)
//
// It is used to:
//  i.  enumerate different link hypotheses in the main tracker, and
//  ii. enumerate different merge/branch hypotheses in the optimiser
//
// types are: LinkHypothesis and MergeHypothesis
template <typename T> class HypothesisMap {
public:
  // default constructor
  HypothesisMap(){};

  // default destructor
  ~HypothesisMap(){};

  // construct a map with n_entries, which are initialised with empty vectors
  // of hypotheses
  HypothesisMap(const unsigned int n_entries) {
    m_hypothesis_map.clear();
    m_hypothesis_map.reserve(n_entries);
    for (size_t i = 0; i < n_entries; i++) {
      m_hypothesis_map.push_back(std::vector<T>());
    }
  };

  // push a new hypothesis into the entry bin
  inline void push(const unsigned int &bin, T lnk) {
    m_hypothesis_map[bin].push_back(lnk);
    m_empty = false;
  };

  // return the number of entries in the HypothesisMap
  size_t size() const { return m_hypothesis_map.size(); };

  // is the container completely empty?
  inline bool empty() const { return m_empty; }

  // return the vector of hypotheses in this bin
  inline std::vector<T> operator[](const unsigned int bin) const {
    // assert(bin < size());
    if (bin >= size())
      return std::vector<T>(); // return empty if no bin exists
    return m_hypothesis_map[bin];
  };

  // count the number of hypotheses in this bin
  const size_t count(const unsigned int &idx) const {
    return m_hypothesis_map[idx].size();
  };

private:
  // the map of hypotheses
  std::vector<std::vector<T>> m_hypothesis_map;

  // empty flag, reset to false if we add a hypothesis
  bool m_empty = true;
};

#endif
