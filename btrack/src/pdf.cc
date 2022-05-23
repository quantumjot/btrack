#include "pdf.h"

using namespace ProbabilityDensityFunctions;
// we can assume that the covar matrix is diagonal from the MotionModel
// since only position obeservations are made, therefore we can decompose
// multivariate gaussian into product of univariate gaussians
// http://cs229.stanford.edu/section/gaussians.pdf

double ProbabilityDensityFunctions::cheat_multivariate_normal(
  const TrackletPtr& trk,
  const TrackObjectPtr& obj
)
{

  Prediction p = trk->predict();
  Eigen::Vector3d x = obj->position();

  double prob_density =

  (1./(kRootTwoPi*sqrt(p.covar(0,0)))) * exp(-(1./(2.*p.covar(0,0))) *
  (x(0)-p.mu(0)) * (x(0)-p.mu(0)) ) *
  (1./(kRootTwoPi*sqrt(p.covar(1,1)))) * exp(-(1./(2.*p.covar(1,1))) *
  (x(1)-p.mu(1)) * (x(1)-p.mu(1)) ) *
  (1./(kRootTwoPi*sqrt(p.covar(2,2)))) * exp(-(1./(2.*p.covar(2,2))) *
  (x(2)-p.mu(2)) * (x(2)-p.mu(2)) );

  return prob_density;

}




// also we need to calculate the probability (the integral), so we use erf
// http://en.cppreference.com/w/cpp/numeric/math/erf

double ProbabilityDensityFunctions::multivariate_erf(
  const TrackletPtr& trk,
  const TrackObjectPtr& obj,
  const double accuracy=2.
)
{

  Prediction p = trk->predict();
  Eigen::Vector3d x = obj->position();

  double phi = 1.;
  double phi_x, std_x, d_x;

  for (unsigned int axis=0; axis<3; axis++) {

    std_x = std::sqrt(p.covar(axis,axis));
    d_x = x(axis)-p.mu(axis);

    // intergral +/- accuracy
    phi_x = std::erf((d_x+accuracy) / (std_x*kRootTwo)) -
            std::erf((d_x-accuracy) / (std_x*kRootTwo));

    // calculate product of integrals for the axes i.e. joint probability?
    phi *= .5*phi_x;

  }

  // we don't want a NaN!
  assert(!std::isnan(phi));

  return phi;
}




// cosine similarity
double ProbabilityDensityFunctions::cosine_similarity(
  const TrackletPtr& trk,
  const TrackObjectPtr& obj
  const double scaling=1.
)
{

  TrackObjectPtr trk_last = trk->track.back();

  // calculate cosine similarity between two feature vectors
  double f_dot = trk_last->features.dot(obj->features);
  double f_mag = (trk_last->features).norm() * (obj->features).norm();
  double cosine_similarity = f_dot / f_mag;

  if (!(cosine_similarity>=-1.0 && cosine_similarity <=1.0)) {
    if (DEBUG) {
      std::cout << cosine_similarity;
      std::cout << " f_dot: " << f_dot;
      std::cout << " f_mag: " << f_mag;
      std::cout << " trk_features: " << (trk_last->features);
      std::cout << " obj_features: " << (obj->features);
      std::cout << std::endl;
    }

    // return a default value
    cosine_similarity = 0.0;
  }

  // rescale to 0.0-1.0
  return (cosine_similarity + 1.0) / 2.0;
}
