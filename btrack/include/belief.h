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

#ifndef _BELIEF_H_INCLUDED_
#define _BELIEF_H_INCLUDED_

#include "math.h"

// Include the CUDA headers if compiling for CUDA
#ifdef __CUDA_ARCH__
#include <cuda.h>
#include <cuda_runtime.h>

// update the belief matrix using a CUDA kernel
void cost_CUDA(float *belief_matrix, const float *new_positions,
               const float *predicted_positions, const unsigned int N,
               const unsigned int T, const float prob_not_assign,
               const float accuracy);
#endif

// Include the Apple Metal headers if compiling for macOS/iOS
#ifdef __APPLE__
#ifdef __cplusplus
extern "C" {
#endif
// update the belief matrix using a Metal kernel
void cost_METAL(float *belief_matrix, const float *new_positions,
                const float *predicted_positions, const unsigned int N,
                const unsigned int T, const float prob_not_assign,
                const float accuracy);
#ifdef __cplusplus
}
#endif
#endif

#endif
