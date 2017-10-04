#ifndef _TYPES_H_INCLUDED_
#define _TYPES_H_INCLUDED_

#include "eigen/Eigen/Dense"
#include <vector>
#include <memory>
#include <iostream>
#include <limits>



// labels for track objects
#define STATE_interphase 101
#define STATE_prometaphase 102
#define STATE_metaphase 103
#define STATE_anaphase 104
#define STATE_apoptosis 105


// errors
#define SUCCESS 900
#define ERROR_empty_queue 901
#define ERROR_no_tracks 902
#define ERROR_no_useable_frames 903
#define ERROR_track_empty 904
#define ERROR_incorrect_motion_model 905
#define ERROR_max_lost_out_of_range 906
#define ERROR_accuracy_out_of_range 907
#define ERROR_prob_not_assign_out_of_range 908
#define ERROR_not_defined 909
#define ERROR_none 910

// constants
const double kInfinity = std::numeric_limits<double>::infinity();



// Python structure for external interface
extern "C" struct PyTrackObject {
  double x;
  double y;
  double z;
  unsigned int t;
  bool dummy;
  unsigned int states;
  unsigned int label;
  double* probability;
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
  PyTrackInfo() : error(ERROR_none), n_tracks(0), n_active(0),
                n_conflicts(0), n_lost(0), t_update_belief(0), t_update_link(0),
                t_total_time(0), p_link(0), p_lost(0), complete(false) {};
};





// TrackObject stores the data of each object in the field of view. Is
// essentially a parallel of the PyTrackObject class.
class TrackObject
{
public:

  // Start a new tracklet without any prior information.
  TrackObject() : x(0.), y(0.), z(0.), t(0), dummy(true), label(0) {};

  // Instantiate a track object from an existing PyTrackObject
  TrackObject(const PyTrackObject& trk) :
              x(trk.x), y(trk.y), z(trk.z), t(trk.t), dummy(trk.dummy),
              label(trk.label), states(trk.states), probability(trk.probability)
              {};

  // Default destructor
  ~TrackObject() {};

  // xyzt position, dummy flag and class label
  double x;
  double y;
  double z;
  unsigned int t;
  bool dummy;
  unsigned int label;
  unsigned int states;
  double* probability;

  // return the current position of the track object
  Eigen::Vector3d position() const {
    Eigen::Vector3d p;
    p << x, y, z;
    return p;
  };



private:

  // Store a reference to the original object?
  // const PyTrackObject* original_object;
};





// Structure for MotionModel predictions
struct Prediction
{
  // position and error predictions
  Eigen::VectorXd mu;
  Eigen::MatrixXd covar;

  // constructors
  // TODO(arl): the default constructor defaults to six states
  Prediction() { mu.setZero(6,1); covar.setIdentity(6,6); }
  Prediction( const Eigen::VectorXd &a_mu,
              const Eigen::MatrixXd &a_covar) : mu(a_mu), covar(a_covar) {}

};


// template <typename btrack_float> struct test_s {
// 	btrack_float x;
// };



// type definition for a track object pointer, minimising copying
typedef std::shared_ptr<TrackObject> TrackObjectPtr;

// Comparison object to order track objects ready for tracking.
inline bool compare_obj_time( const TrackObjectPtr trackobj_1,
                              const TrackObjectPtr trackobj_2 ) {
  return (trackobj_1->t < trackobj_2->t);
}

// Structure to determine the imaging volume from the observations
struct ImagingVolume
{
  // minimum and maximum dimensions of the volume from observations
  Eigen::Vector3d min_xyz;
  Eigen::Vector3d max_xyz;

  // default constructor
  ImagingVolume() { min_xyz.fill(kInfinity); max_xyz.fill(-kInfinity); };

  // update with an observation
  void update(const TrackObjectPtr obj) {
    min_xyz = min_xyz.cwiseMin( obj->position() );
    max_xyz = max_xyz.cwiseMax( obj->position() );
  };

  // test whether an object lies within the volume
  bool inside(const Eigen::Vector3d& position) const {
  if (position(0)>= min_xyz(0) && position(0)<=max_xyz(0) &&
      position(1)>= min_xyz(1) && position(1)<=max_xyz(1) &&
      position(2)>= min_xyz(2) && position(2)<=max_xyz(2)) {
        return true;
      } else {
        return false;
      }
  }
};

#endif
