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

#ifndef _TRACKER_H_INCLUDED_
#define _TRACKER_H_INCLUDED_

#include "eigen/Eigen/Dense"
//#include <python2.7/Python.h>
#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <limits>
#include <algorithm>
#include <set>
#include <ctime>

#include "types.hpp"
#include "motion.hpp"
#include "belief.h"
#include "inference.hpp"

#define MAX_LOST 5
#define PROB_NOT_ASSIGN 0.01
#define DEFAULT_ACCURACY 2.0
#define PROB_ASSIGN_EXP_DECAY true
#define DYNAMIC_ACCURACY false
#define DIMS 3
#define DEBUG true


// reserve space for objects and tracks
#define RESERVE_NEW_OBJECTS 1000
#define RESERVE_ALL_TRACKS 500000
#define RESERVE_ACTIVE_TRACKS 1000


// constants for integrating Gaussian PDF
const double kRootTwo = std::sqrt(2.0);
const double kRootTwoPi = std::sqrt(2.0*M_PI);






// Tracklet object. A container class to keep the list of track objects as well
// as a dedicated motion and object models for the object.
class Tracklet
{
public:
  // default constructor for Tracklet
  Tracklet() {};

  // construct Tracklet using a new ID, new object and model specific parameters
  Tracklet( const unsigned int new_ID,
            const TrackObjectPtr new_object,
            const unsigned int max_lost,
            const MotionModel& model );

  // default destructor for Tracklet
  ~Tracklet() {};

  // append a new track object to the trajectory, update flag tells the function
  // whether to update the motion model or not - new tracks should not update
  // the motion model
  void append(const TrackObjectPtr new_object, bool update);

  // append a dummy object to the trajectory in case of a missed observation
  void append_dummy();

  // return the length of the trajectory
  unsigned int length() const { return track.size(); };

  // return a boolean representing the status (active/inactive) of the track
  bool active() const { return lost<max_lost; };

  // trim trailing dummy objects - should only be called when the tracking is
  // finished
  bool trim();

  // get the track data as a C-type array
  // TODO(arl): implement this
  double* get();

  // get the position coordinates over time
  // TODO(arl): implement these
  std::vector<float> x();
  std::vector<float> y();
  std::vector<float> z();
  std::vector<unsigned int> t();
  std::vector<bool> dummy();

  // get the current position from the last known object
  Eigen::Vector3d position() const {
    return track.back()->position();
  };

  // set the track to lost - permanently!
  void set_lost() {
    lost = max_lost+1;
  }


  // get the latest prediction from the motion model. Note that the current
  // prediction from the Tracklet object is different to the prediction of the
  // motion model. The tracklet adds any extra model information to the
  // last known position, while the motion model is the filtered version of the
  // data which may contain some lag. This is a critical part of the prediction
  // TODO(arl): make this model agnostic
  Prediction predict() const;

  // Identifier for the tracklet
  unsigned int ID = 0;

  // these are vectors storing the predicted new position and the Kalman output
  // as well as the pointers to the track objects comprising the trajectory
  std::vector<Prediction> kalman;
  std::vector<Prediction> prediction;
  std::vector<TrackObjectPtr> track;

  // counter for number of consecutive lost/dummy observations
  unsigned int lost = 0;

private:

  // if the lost counter exceeds max_lost, the track is considered inactive
  unsigned int max_lost = MAX_LOST;

  // motion model
  MotionModel motion_model;

  // object model
  ObjectModel object_model;
};



// a shared pointer for tracklets
typedef std::shared_ptr<Tracklet> TrackletPtr;

// a pair for hypotheses, track/object ID and probability
typedef std::pair<unsigned int, double> LinkHypothesis;





// A HypothesisMap container that maps track->object link hypotheses. This can
// be used to determine whether linking conflicts exist and the posterior
// probability of each one.
class HypothesisMap
{
public:
  // default constructor
  HypothesisMap() {};

  // default destructor
  ~HypothesisMap() {};

  // construct a map with n_entries, which are initialised with empty vectors
  // of hypotheses
  HypothesisMap(const unsigned int n_entries){
    trackmap.reserve(n_entries);
    for (size_t i=0; i<n_entries; i++) {
      trackmap.push_back( std::vector<LinkHypothesis>() );
    }
  };

  // push a new hypothesis into the entry bin
  inline void push(const unsigned int &idx, LinkHypothesis lnk) {
    trackmap[idx].push_back(lnk);
  };

  // return the number of entries in the HypothesisMap
  size_t size() const {
    return trackmap.size();
  };

  // return the vector of hypotheses in this bin
  inline std::vector<LinkHypothesis> operator[] (const unsigned int idx) const {
    return trackmap[idx];
  };

  // count the number of hypotheses in this bin
  const size_t count(const unsigned int &idx) const {
    return trackmap[idx].size();
  };

private:
  // the map of hypotheses
  std::vector< std::vector<LinkHypothesis> > trackmap;
};















// BayesianTracker is a multi object tracking algorithm, specifically
// used to reconstruct tracks in crowded fields. Here we use a probabilistic
// network of information to perform the trajectory linking. This method uses
// positional information (position, velocity ...) as well as visual
// information (intensity, correlation...) for track linking.
//
// The tracking algorithm assembles reliable sections of track that do not
// contain cell division events (tracklets). Each new tracklet initiates a
// probabilistic model in the form of a Kalman filter (Kalman, 1960), and
// utilises this to predict future states (and error in states) of each of the
// objects in the field of view.  We assign new observations to the growing
// tracklets (linking) by evaluating the posterior probability of each
// potential linkage from a Bayesian belief matrix for all possible linkages
// (Narayana and Haverkamp, 2007). The best linkages are those with the highest
// posterior probability.

class BayesianTracker
{
public:
  // Default constructor
  BayesianTracker();
  BayesianTracker(bool verbose);

  // Default destructor
  ~BayesianTracker();

  // set some parameters
  // TODO(arl): is this essential anymore?
  // unsigned int setup(const double prob_not_assign, const unsigned int max_lost,
  // 									const double accuracy);

  // set up the motion model. matrices are in the form of c-style linear arrays
  unsigned int set_motion_model(const unsigned int measurements,
                                const unsigned int states,
                                double* A_raw,
                                double* H_raw,
                                double* P_raw,
                                double* Q_raw,
                                double* R_raw,
                                const double dt,
                                const double accuracy,
                                const unsigned int max_lost,
                                const double prob_not_assign);

  // set up the object model
  unsigned int set_object_model(const unsigned int states,
                                double* transition_raw,
                                double* emission_raw,
                                double* start_raw);

  // add new objects
  unsigned int xyzt(const double* xyzt);
  unsigned int append(const PyTrackObject& new_object);

  // infer the volume of observations
  void infer_tracking_volume() const;

  // initialise the tracker with the first observations
  unsigned int initialise();

  // track and update tracks NOTE: this will be deprecated
  // unsigned int track(const unsigned int end_frame);

  // run the tracking on the entire set
  void track_all();

  // move the tracking forward by n iterations, used in interactive mode
  void step() { step(1); };
  void step(const unsigned int n_steps);

  // get the number of tracks
  inline unsigned int size() const {
    return tracks.size();
  };

  // return the Euclidean distance between object and trajectory
  double euclidean_dist(const size_t trk, const size_t obj) const {
    Eigen::Vector3d dxyz = tracks[trk]->position()-new_objects[obj]->position();
    return std::sqrt(dxyz.transpose()*dxyz);
  };

  // track maintenance
  bool clean();
  bool renumber();
  bool purge();

  // calculate the cost matrix
  void cost(Eigen::Ref<Eigen::MatrixXd> belief,
            const size_t n_tracks,
            const size_t n_objects);

  // calculate linkages based on belief matrix
  void link(Eigen::Ref<Eigen::MatrixXd> belief,
            const size_t n_tracks,
            const size_t n_objects);

  // somewhere to store the tracks
  std::vector<TrackletPtr> tracks;

  // maintain the size of the ImagingVolume
  ImagingVolume volume;

  // statistics
  const PyTrackInfo* stats() {
    return &statistics;
  }

private:

  // verbose output to stdio
  bool verbose = false;

  // default motion model, must remain uninitialised
  MotionModel motion_model;

  // default object model, must remain uninitialised
  ObjectModel object_model;

  // default tracking parameters
  double prob_not_assign = PROB_NOT_ASSIGN;
  double accuracy = DEFAULT_ACCURACY;
  unsigned int max_lost = MAX_LOST;

  // provide a global ID counter for new tracks
  inline unsigned int get_new_ID() {
    new_ID++; return new_ID;
  };

  // display the debug output to std::out
  void debug_output(const unsigned int frm) const;

  // update the list of active tracks
  bool update_active();

  // maintain pointers to tracks
  std::vector<TrackletPtr> active;
  std::vector<TrackObjectPtr> new_objects;

  // some space to store the objects
  std::vector<TrackObjectPtr> objects;

  // sizes of various vectors
  size_t n_objects;

  unsigned int current_frame;
  unsigned int o_counter;

  // store the frame numbers of incoming tracks
  std::set<unsigned int> frames_set;
  std::vector<unsigned int> frames;

  // tracker initialisation
  bool initialised = false;

  // ID counter for new tracks
  unsigned int new_ID = 0;

  // counter to run the purge function
  unsigned int purge_iter;

  // counters for the number of lost tracks and number of conflicts
  unsigned int n_lost = 0;
  unsigned int n_conflicts = 0;

  // set up a structure for the statistics
  PyTrackInfo statistics;
};







#endif
