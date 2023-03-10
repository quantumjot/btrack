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

#ifndef _PDF_H_INCLUDED_
#define _PDF_H_INCLUDED_

#include "eigen/Eigen/Dense"
#include <vector>
#include <iostream>
#include <cmath>

#include "defs.h"
#include "types.h"
#include "tracklet.h"

// #pragma once
namespace ProbabilityDensityFunctions
{
  double cheat_multivariate_normal(
    const TrackletPtr& trk,
    const TrackObjectPtr& obj
  );

  double multivariate_erf(
    const TrackletPtr& trk,
    const TrackObjectPtr& obj,
    const double accuracy
  );

  double cosine_similarity(
    const TrackletPtr& trk,
    const TrackObjectPtr& obj
  );

  double soft_cosine_similarity(
    const TrackletPtr& trk,
    const TrackObjectPtr& obj
  );
}

#endif
