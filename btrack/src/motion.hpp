#ifndef _MOTION_H_INCLUDED_
#define _MOTION_H_INCLUDED_

#include "eigen/Eigen/Dense"
#include "types.hpp"

#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <limits>
#include <algorithm>
#include <set>

// Implements a Kalman filter for motion modelling in the tracker. Note that we
// do not implement the 'control' updates from the full Kalman filter
class MotionModel
{
public:
	// Default constructor for MotionModel
	MotionModel() {};

	// Initialise a motion model with matrices representing the motion model.
	// A: State transition matrix
	// H: Observation matrix
	// P: Initial covariance estimate
	// Q: Estimated error in process
	// R: Estimated error in measurements
	// Certain parameters are inferred from the shapes of the matrices, such as
	// the number of states and measurements
	MotionModel(const Eigen::MatrixXd &A,
	 						const Eigen::MatrixXd &H,
						 	const Eigen::MatrixXd &P,
						 	const Eigen::MatrixXd &R,
						 	const Eigen::MatrixXd &Q);

	// Default destructor
	~MotionModel() {};

	// Setup the filter from a new object, set x_hat to the object position
	void setup(const TrackObjectPtr new_object);

	// run an update of the Kalman filter using a new object observation
	void update(const TrackObjectPtr new_object);

	// get the Kalman filter prediction
	Prediction predict() const;

	// return the system dimensions
	void dimensions(unsigned int* m,
									unsigned int* s) const {
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

	// an identity matrix
	Eigen::MatrixXd I;

	// initialised is set to true by the constructor
	bool initialised = false;

	// time step which will default to one
	double dt;
};

#endif
