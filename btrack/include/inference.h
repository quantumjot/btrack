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

#ifndef _INFERENCE_H_INCLUDED_
#define _INFERENCE_H_INCLUDED_

#include "eigen/Eigen/Dense"
#include "types.h"

#include <vector>

#define RESERVE_STATE_SEQUENCE 1000

class ObjectModel
{
  public:
    ObjectModel() {};
    ~ObjectModel() {};

    // initialise with transition, emission and start matrices
    ObjectModel(const Eigen::MatrixXd &transition,
                const Eigen::MatrixXd &emission,
                const Eigen::MatrixXd &start);

    // initialise with a uniform prior
    ObjectModel(const Eigen::MatrixXd &transition,
                const Eigen::MatrixXd &emission);

    // add a new observation to the chain
    void update(const TrackObjectPtr new_object);

    // make a prediction from the model
    Eigen::VectorXd predict();

  private:

    // the emission and transition matrices
    Eigen::MatrixXd transition;
    Eigen::MatrixXd emission;

    // store the number of states of the system
    unsigned int states;

    // space to store the current state
    Eigen::VectorXd x_hat;

    // store the state sequence
    std::vector<unsigned short> sequence;

    // make a prediction based on the model
    void forward(const unsigned int observation);

    // backward algorithm
    void backward();
};

#endif
