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

#include "inference.h"

ObjectModel::ObjectModel(const Eigen::MatrixXd &transition,
            const Eigen::MatrixXd &emission,
            const Eigen::MatrixXd &start) : transition(transition),
            emission(emission), states(transition.rows()), x_hat(start)
{
  sequence.reserve(RESERVE_STATE_SEQUENCE);
}



ObjectModel::ObjectModel(const Eigen::MatrixXd &transition,
            const Eigen::MatrixXd &emission) : transition(transition),
            emission(emission), states(transition.rows())
{
  // set up the start state with a uniform prior
  x_hat(states);
  x_hat.fill(1.0/(double)states);;

  sequence.reserve(RESERVE_STATE_SEQUENCE);
}



// run the forward pass of the model
void ObjectModel::forward(const unsigned int observation) {
  Eigen::VectorXd update(states);
  for (unsigned int s=0; s<states; s++) {
    update = transition.col(s).array() * x_hat.array();
    x_hat(s) = emission(s, observation) * update.sum();
  }
}



// run the backward pass of the model
void ObjectModel::backward() {
}



// update with a new observation
void ObjectModel::update(const TrackObjectPtr new_object) {
  // push back the new observation and update predictions
  sequence.push_back(new_object->label);
  forward( new_object->label );
}



Eigen::VectorXd ObjectModel::predict() {
  // make a prediction based on the transition matrix
  Eigen::VectorXd prediction(states);
  prediction = x_hat * transition;
  return prediction;
}
