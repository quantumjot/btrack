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

#ifndef _TRACKLET_H_INCLUDED_
#define _TRACKLET_H_INCLUDED_

#include <vector>

#include "defs.h"
#include "eigen/Eigen/Dense"
#include "inference.h"
#include "motion.h"
#include "types.h"

// Tracklet object. A container class to keep the list of track objects as well
// as a dedicated motion and object models for the object.
class Tracklet {
public:
  // default constructor for Tracklet
  Tracklet() : remove_flag(false){};

  // construct Tracklet using a new ID, new object and model specific
  // parameters
  Tracklet(const unsigned int new_ID, const TrackObjectPtr &new_object,
           const unsigned int max_lost, const MotionModel &model);

  // default destructor for Tracklet
  ~Tracklet(){};

  // append a new track object to the trajectory, update flag tells the
  // function whether to update the motion model or not - new tracks should
  // not update the motion model
  void append(const TrackObjectPtr &new_object, bool update);
  void append(const TrackObjectPtr &new_object) { append(new_object, true); }
  // append a dummy object to the trajectory in case of a missed observation
  // only update the predicted position in cases where we're using a motion
  // model, default with no args is to update the position
  void append_dummy(const bool update_position);
  void append_dummy() { append_dummy(true); };

  // return the length of the trajectory
  unsigned int length() const { return track.size(); };

  // return the track duration
  double duration() const {
    assert(!track.empty());
    return track.back()->t - track.front()->t;
  }

  // return a boolean representing the status (active/inactive) of the track
  bool active() const { return lost <= max_lost; };

  // trim trailing dummy objects - should only be called when the tracking is
  // finished
  bool trim();

  // get the track data as a C-type array
  // TODO(arl): implement this
  double *get();

  // get the position coordinates over time
  // TODO(arl): implement these
  std::vector<float> x();
  std::vector<float> y();
  std::vector<float> z();
  std::vector<unsigned int> t();
  std::vector<bool> dummy();

  // get the current position from the last known object
  Eigen::Vector3d position() const {
    assert(!track.empty());
    return track.back()->position();
  };

  // set the track to lost - permanently!
  void set_lost() { lost = max_lost + 1; }

  // check to see whether this should be removed;
  bool to_remove() const { return remove_flag; }

  // set a flag to remove this track
  void to_remove(bool a_remove) { remove_flag = a_remove; }

  // does this track have children
  bool has_children() const { return (child_one != child_two); }

  // get the latest prediction from the motion model. Note that the current
  // prediction from the Tracklet object is different to the prediction of the
  // motion model. The tracklet adds any extra model information to the
  // last known position, while the motion model is the filtered version of
  // the data which may contain some lag. This is a critical part of the
  // prediction
  // TODO(arl): make this model agnostic
  Prediction predict() const;

  // Identifier for the tracklet
  unsigned int ID = 0;

  // these are vectors storing the predicted new position and the Kalman
  // output as well as the pointers to the track objects comprising the
  // trajectory
  std::vector<Prediction> kalman;
  std::vector<Prediction> prediction;
  std::vector<TrackObjectPtr> track;

  // counter for number of consecutive lost/dummy observations
  unsigned int lost = 0;

  // store the root, parent and original IDs
  unsigned int root = 0;
  unsigned int parent = 0;
  unsigned int child_one = 0;
  unsigned int child_two = 0;
  unsigned int renamed_ID;
  unsigned int fate = TYPE_undef;
  unsigned int generation = 0;

private:
  // if the lost counter exceeds max_lost, the track is considered inactive
  unsigned int max_lost = MAX_LOST;

  // set the remove flag
  bool remove_flag = false;

  // motion model
  MotionModel motion_model;

  // object model
  ObjectModel object_model;
};

// a shared pointer for tracklets
typedef std::shared_ptr<Tracklet> TrackletPtr;

#endif
