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

#ifndef _FEATURES_H_INCLUDED_
#define _FEATURES_H_INCLUDED_

#include "defs.h"



inline bool _use_features(const unsigned int features, const unsigned int bitmask)
{
  unsigned int bitmask_motion = std::pow(2, bitmask);
  return ((features & bitmask) == bitmask);
};


// deals with setting which features to use during tracking or hypothesis
// generation. This class is inherited by both BayesianTracker and
// HypothesisEngine to provide a common interface

class UpdateFeatures
{
  public:

    // set the features to use while updating the Bayesian belief matrix
    inline void set_update_features(const unsigned int update_features) {
      m_update_features = update_features;
    };

    inline const unsigned int get_update_features(void) const {
      return m_update_features;
    };

    inline bool use_motion_features(void) const {
      return _use_features(m_update_features, USE_MOTION_FEATURES);
    };

    inline bool use_visual_features(void) const {
      return _use_features(m_update_features, USE_VISUAL_FEATURES);
    };


  protected:
    unsigned int m_update_features;

  private:
    //

};

#endif
