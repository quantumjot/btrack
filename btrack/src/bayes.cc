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

#include "bayes.h"

using namespace BayesianUpdateFunctions;

std::tuple<double, double> BayesianUpdateFunctions::safe_bayesian_update(
    double prior_assign, double prob_assign, double prob_not_assign) {
  double safe_prior_assign = std::max(prior_assign, 1e-10);
  double inv_prior_assign = 1.0 - safe_prior_assign;
  // Fix: Use inv_prior_assign instead of (1. - prob_assign)
  // Correct Bayes' rule: P(D) = P(D|H)*P(H) + P(D|¬H)*P(¬H)
  double PrDP =
      prob_assign * safe_prior_assign + prob_not_assign * inv_prior_assign;

  // Prevent division by zero
  if (PrDP < 1e-15) {
    PrDP = 1e-15;
  }

  double posterior = (prob_assign * safe_prior_assign) / PrDP;

  // Prevent division by zero in update calculation
  double update =
      (inv_prior_assign > 1e-15)
          ? (1. + (safe_prior_assign - posterior) / inv_prior_assign)
          : 1.0;

  return std::make_tuple(update, posterior);
}

double BayesianUpdateFunctions::safe_bayesian_update_simple(double prior,
                                                            double likelihood) {
  double posterior;
  double safe_prior = std::max(prior, 1e-99);
  posterior =
      1. / (1. + ((1. / safe_prior) - 1.) * ((1. - safe_prior) / likelihood));

  return posterior;
}
