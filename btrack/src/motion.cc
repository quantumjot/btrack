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

#include "motion.h"

MotionModel::MotionModel( const Eigen::MatrixXd &A,
                          const Eigen::MatrixXd &H,
                          const Eigen::MatrixXd &P,
                          const Eigen::MatrixXd &R,
                          const Eigen::MatrixXd &Q ) :
                          A(A), H(H), P(P), R(R), Q(Q),
                          measurements(H.rows()), states(A.rows()),
                          x_hat(states), x_hat_new(states), I(states,states)
{

  // set the time step and the identity matrix used by the Kalman filter
  dt = 1;
  I.setIdentity();

  // fill the current prediction and motion vector with zeros
  x_hat.fill(0.0);
  motion_vector.fill(0.0);

  // set an initialised flag
  initialised = true;
}



// setup with a new observation
// DONE(arl): make this agnostic to model
void MotionModel::setup(const TrackObjectPtr new_object)
{
  x_hat.head(3) = new_object->position();
};



// return a prediction of the position and (co)variance
Prediction MotionModel::predict() const
{
  assert(initialised);
  Prediction p = Prediction(x_hat, P);
  return p;
}



  // update the model with a new observation or dummy
void MotionModel::update(const TrackObjectPtr new_object)
{
  assert(initialised);

  // discrete Kalman filter time update, no control...
  x_hat_new = A * x_hat;
  P = A*P*A.transpose() + Q;

  // if this is a dummy object, end here. Update prediction without new data
  if (new_object->dummy) {
    x_hat = x_hat_new;
    return;
  }

  // kalman gain and the measurement update
  K = P*H.transpose()*(H*P*H.transpose() + R).inverse();
  x_hat_new = x_hat_new + K * (new_object->position() - H*x_hat_new);

  // update the motion vector, essentially the difference in position
  motion_vector = x_hat_new.head(3) - x_hat.head(3);

  // update P and the predicted state
  P = (I - K*H)*P;
  x_hat = x_hat_new;
}
