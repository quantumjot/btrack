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

#ifndef _BAYES_H_INCLUDED_
#define _BAYES_H_INCLUDED_

#include <algorithm>
#include <cmath>
#include <tuple>

namespace BayesianUpdateFunctions {
std::tuple<double, double> safe_bayesian_update(double prior_assign,
                                                double prob_assign,
                                                double prob_not_assign);

double safe_bayesian_update_simple(double prior, double likelihood);
} // namespace BayesianUpdateFunctions

#endif
