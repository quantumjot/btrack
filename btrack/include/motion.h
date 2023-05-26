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

#ifndef _MOTION_H_INCLUDED_
#define _MOTION_H_INCLUDED_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include "eigen/Eigen/Dense"
#include "types.h"

// Implements a Kalman filter for motion modelling in the tracker. Note that we
// do not implement the 'control' updates from the full Kalman filter
class MotionModel {
public:
  // Default constructor for MotionModel
  MotionModel(){};

  // Initialise a motion model with matrices representing the motion model.
  // A: State transition matrix
  // H: Observation matrix
  // P: Initial covariance estimate
  // Q: Estimated error in process
  // R: Estimated error in measurements
  // Certain parameters are inferred from the shapes of the matrices, such as
  // the number of states and measurements
  MotionModel(const Eigen::MatrixXd &A, const Eigen::MatrixXd &H,
              const Eigen::MatrixXd &P, const Eigen::MatrixXd &R,
              const Eigen::MatrixXd &Q);

  // Default destructor
  ~MotionModel(){};

  // Setup the filter from a new object, set x_hat to the object position
  void setup(const TrackObjectPtr new_object);

  // run an update of the Kalman filter using a new object observation
  void update(const TrackObjectPtr new_object);

  // get the Kalman filter prediction
  Prediction predict() const;

  // get the motion vector
  Eigen::Vector3d get_motion_vector() const { return motion_vector; }

  // return the system dimensions
  void dimensions(unsigned int *m, unsigned int *s) const {
    *m = measurements;
    *s = states;
  };

private:
  // matrices for Kalman filter
  Eigen::MatrixXd A;
  Eigen::MatrixXd H;
  Eigen::MatrixXd P;
  Eigen::MatrixXd R;
  Eigen::MatrixXd Q;
  Eigen::MatrixXd K;

  // system dimensions
  unsigned int measurements;
  unsigned int states;

  // store states
  Eigen::VectorXd x_hat;
  Eigen::VectorXd x_hat_new;

  // motion vector
  Eigen::Vector3d motion_vector;

  // an identity matrix
  Eigen::MatrixXd I;

  // initialised is set to true by the constructor
  bool initialised = false;

  // time step which will default to one
  double dt;
};

#endif
