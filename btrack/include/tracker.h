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

#ifndef _TRACKER_H_INCLUDED_
#define _TRACKER_H_INCLUDED_

#include "Eigen/Dense"
#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <limits>
#include <algorithm>
#include <set>
#include <ctime>
#include <fstream>

#include "types.h"
#include "motion.h"
#include "inference.h"
#include "tracklet.h"
#include "manager.h"
#include "defs.h"
#include "hyperbin.h"
#include "pdf.h"
#include "updates.h"
#include "bayes.h"


// #define PROB_NOT_ASSIGN 0.01
// #define DEFAULT_ACCURACY 2.0
// #define DISALLOW_METAPHASE_ANAPHASE_LINKING true
// #define PROB_ASSIGN_EXP_DECAY true
// #define DYNAMIC_ACCURACY false
// #define DIMS 3
// #define FAST_COST_UPDATE false
//
//
// // reserve space for objects and tracks
// #define RESERVE_NEW_OBJECTS 1000
// #define RESERVE_ACTIVE_TRACKS 1000










// a pair for hypotheses, track/object ID and probability
typedef std::pair<unsigned int, double> LinkHypothesis;



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

class BayesianTracker: public UpdateFeatures
{
public:
  // Constructor
  BayesianTracker() {};
  // BayesianTracker(const bool verbose);
  BayesianTracker(const bool verbose,
                  const unsigned int update_mode);

  // Default destructor
  ~BayesianTracker();

  // set some parameters
  // TODO(arl): is this essential anymore?
  // unsigned int setup(const double prob_not_assign, const unsigned int max_lost,
  //                                    const double accuracy);

  // set the cost function to use
  void set_update_mode(const unsigned int update_mode);

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

  // set the maximum search radius
  void set_max_search_radius(const float search_radius) {
    this->max_search_radius = search_radius;
  }

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

  // // return the Euclidean distance between object and trajectory
  // double euclidean_dist(const size_t trk, const size_t obj) const {
  //   Eigen::Vector3d dxyz = tracks[trk]->position()-new_objects[obj]->position();
  //   return std::sqrt(dxyz.transpose()*dxyz);
  // };

  double euclidean_dist(const TrackletPtr& trk, const TrackObjectPtr& obj) const {
    Eigen::Vector3d dxyz = trk->position() - obj->position();
    return std::sqrt(dxyz.transpose()*dxyz);
  };


  // track maintenance
  bool purge();

  // calculate the cost matrix using different methods
  void cost_EXACT(Eigen::Ref<Eigen::MatrixXd> belief,
                  const size_t n_tracks,
                  const size_t n_objects,
                  const bool use_uniform_prior);

  void cost_APPROXIMATE(Eigen::Ref<Eigen::MatrixXd> belief,
                        const size_t n_tracks,
                        const size_t n_objects,
                        const bool use_uniform_prior);

  void cost_CUDA(Eigen::Ref<Eigen::MatrixXd> belief,
                 const size_t n_tracks,
                 const size_t n_objects,
                 const bool use_uniform_prior);

  // calculate linkages based on belief matrix
  void link(Eigen::Ref<Eigen::MatrixXd> belief,
            const size_t n_tracks,
            const size_t n_objects);


  double prob_update_motion(const TrackletPtr& trk, const TrackObjectPtr& obj) const;
  double prob_update_visual(const TrackletPtr& trk, const TrackObjectPtr& obj) const;

  // somewhere to store the tracks
  TrackManager tracks;

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

  // cost function mode
  // NOTE(arl): this is probably obsolete now
  unsigned int cost_function_mode;

  // reference to an update function
  double (BayesianTracker::*m_update_fn)(
    const TrackletPtr&,
    const TrackObjectPtr&
  ) const;

  // reference to a cost function
  void (BayesianTracker::*m_cost_fn)(
    const Eigen::Ref<Eigen::MatrixXd>,
    const size_t,
    const size_t,
    const bool
  );

  // default tracking parameters
  double prob_not_assign = PROB_NOT_ASSIGN;
  double accuracy = DEFAULT_ACCURACY;
  unsigned int max_lost = MAX_LOST;

  // provide a global ID counter for new tracks
  // NOTE(arl): because the increment is before the return, all tracks will
  // be numbered 1 and upward.
  inline unsigned int get_new_ID() {
    new_ID++; return new_ID;
  };

  // display the debug output to std::out
  void debug_output(const unsigned int frm) const;

  // update the list of active tracks
  bool update_active();

  // pointer to the track manager
  // TrackManager* p_manager;

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
  float max_search_radius = MAX_SEARCH_RADIUS;

  // set up a structure for the statistics
  PyTrackInfo statistics;
};



// utils to write out belief matrix to CSV files
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,
                                       Eigen::DontAlignCols, ", ", "\n");

void write_belief_matrix_to_CSV(std::string a_filename,
                                Eigen::Ref<Eigen::MatrixXd> a_belief);

#endif
