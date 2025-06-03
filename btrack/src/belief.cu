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

// TODO(arl): Work in Progress...

#include "belief.h"

// const float PI_C = 3.141592653589793238463;
const float ROOT_TWO = 1.4142135623730951;

__device__ float probability_erf_CUDA(const float x, const float y,
                                      const float z, const float px,
                                      const float py, const float pz,
                                      const float var_x, const float var_y,
                                      const float var_z, const float accuracy) {

  float phi_x, std_x;
  float phi_y, std_y;
  float phi_z, std_z;

  std_x = sqrtf(var_x);
  std_y = sqrtf(var_y);
  std_z = sqrtf(var_z);

  // intergral x
  phi_x = erff((x - px + accuracy) / (std_x * ROOT_TWO)) -
          erff((x - px - accuracy) / (std_x * ROOT_TWO));

  // intergral y
  phi_y = erff((y - py + accuracy) / (std_y * ROOT_TWO)) -
          erff((y - py - accuracy) / (std_y * ROOT_TWO));

  // intergral z
  phi_z = erff((z - pz + accuracy) / (std_z * ROOT_TWO)) -
          erff((z - pz - accuracy) / (std_z * ROOT_TWO));

  // joint probability
  float phi = .5 * phi_x * .5 * phi_y * .5 * phi_z;

  // calculate product of integrals for the axes i.e. joint probability?
  return phi;
}

__global__ void track_CUDA(float *belief_matrix, const float *new_positions,
                           const float *predicted_positions,
                           const unsigned int num_obj,
                           const unsigned int num_tracks,
                           const float prob_not_assign, const float accuracy) {
  /* perform the acutal tracking association step */

  // this refers to the active track
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  // this refers to the new objects
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  // Don't bother doing the calculation. We're not in a valid location
  if (i >= num_tracks || j >= num_obj)
    return;
  // NOTE: we need to update the lost column

  // set up the probability for assignment
  float prob_assign = 0.;

  // first, get the value of the belief matrix in its current step
  float prior_assign = belief_matrix[j * num_tracks + i];

  float x, y, z, px, py, pz, var_x, var_y, var_z;

  // get the new object positions
  x = new_positions[j * 3 + 0];
  y = new_positions[j * 3 + 1];
  z = new_positions[j * 3 + 2];

  // get the predictions from the tracks
  px = predicted_positions[i * 6 + 0];
  py = predicted_positions[i * 6 + 1];
  pz = predicted_positions[i * 6 + 2];
  var_x = predicted_positions[i * 6 + 3];
  var_y = predicted_positions[i * 6 + 4];
  var_z = predicted_positions[i * 6 + 5];

  // get the probability of assignment for this track
  prob_assign =
      probability_erf_CUDA(x, y, z, px, py, pz, var_x, var_y, var_z, accuracy);
  float PrDP =
      prob_assign * prior_assign + prob_not_assign * (1. - prob_assign);
  float posterior = (prob_assign * (prior_assign / PrDP));

  // update the posterior for this track
  belief_matrix[j * num_tracks + i] = posterior;
}

// This is the interface to the outside world!
void cost_CUDA(float *belief_matrix, const float *new_positions,
               const float *predicted_positions, const unsigned int N,
               const unsigned int T, const float prob_not_assign,
               const float accuracy) {

  // allocate some device memory
  float *belief, *new_pos, *pred_pos;
  cudaMalloc(&belief, N * T * sizeof(float));
  cudaMalloc(&new_pos, N * 3 * sizeof(float));
  cudaMalloc(&pred_pos, T * 6 * sizeof(float));

  // copy to the GPU
  cudaMemcpy(belief, belief_matrix, N * T * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(new_pos, new_positions, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(pred_pos, predicted_positions, T * 6 * sizeof(float),
             cudaMemcpyHostToDevice);

  // execute the kernel with proper 2D grid configuration
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (T + block.y - 1) / block.y);
  track_CUDA<<<grid, block>>>(belief, new_pos, pred_pos, N, T, prob_not_assign,
                       accuracy);

  // get the belief matrix back
  cudaMemcpy(belief_matrix, belief, N * T * sizeof(float),
             cudaMemcpyDeviceToHost);

  // clean up
  cudaFree(belief);
  cudaFree(new_pos);
  cudaFree(pred_pos);
}

int main_CUDA(void) {
  // do nothing - we'll call this from elsewhere
  return 0;

}
