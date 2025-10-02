/*
--------------------------------------------------------------------------------
 Name:     belief.metal
 Purpose:  Metal compute shader for belief matrix updates
 Authors:  Based on CUDA implementation by Alan R. Lowe (arl) a.lowe@ucl.ac.uk
 License:  See LICENSE.md
--------------------------------------------------------------------------------
*/

#include <metal_stdlib>
using namespace metal;

constant float ROOT_TWO = 1.4142135623730951;

float erf(float x) {
    const float a1 =  0.254829592;
    const float a2 = -0.284496736;
    const float a3 =  1.421413741;
    const float a4 = -1.453152027;
    const float a5 =  1.061405429;
    const float p  =  0.3275911;

    int sign = (x < 0) ? -1 : 1;
    x = fabs(x);

    float t = 1.0 / (1.0 + p * x);
    float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return sign * y;
}


float probability_erf_Metal(const float x, const float y, const float z,
                            const float px, const float py, const float pz,
                            const float var_x, const float var_y, const float var_z,
                            const float accuracy) {
    float phi_x, std_x;
    float phi_y, std_y;
    float phi_z, std_z;
    
    std_x = sqrt(var_x);
    std_y = sqrt(var_y);
    std_z = sqrt(var_z);
    
    // integral x
    phi_x = erf((x - px + accuracy) / (std_x * ROOT_TWO)) -
            erf((x - px - accuracy) / (std_x * ROOT_TWO));
    
    // integral y
    phi_y = erf((y - py + accuracy) / (std_y * ROOT_TWO)) -
            erf((y - py - accuracy) / (std_y * ROOT_TWO));
    
    // integral z
    phi_z = erf((z - pz + accuracy) / (std_z * ROOT_TWO)) -
            erf((z - pz - accuracy) / (std_z * ROOT_TWO));
    
    // joint probability
    float phi = 0.5 * phi_x * 0.5 * phi_y * 0.5 * phi_z;
    
    return phi;
}

kernel void track_Metal(device float* belief_matrix [[buffer(0)]],
                       device const float* new_positions [[buffer(1)]],
                       device const float* predicted_positions [[buffer(2)]],
                       constant uint& num_obj [[buffer(3)]],
                       constant uint& num_tracks [[buffer(4)]],
                       constant float& prob_not_assign [[buffer(5)]],
                       constant float& accuracy [[buffer(6)]],
                       uint2 gid [[thread_position_in_grid]]) {
    
    // this refers to the active track
    uint i = gid.y;
    // this refers to the new objects
    uint j = gid.x;
    
    // Don't bother doing the calculation. We're not in a valid location
    if (i >= num_tracks || j >= num_obj)
        return;
    
    // set up the probability for assignment
    float prob_assign = 0.0;
    
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
    prob_assign = probability_erf_Metal(x, y, z, px, py, pz, var_x, var_y, var_z, accuracy);
    float PrDP = prob_assign * prior_assign + prob_not_assign * (1.0 - prob_assign);
    float posterior = (prob_assign * (prior_assign / PrDP));
    
    // update the posterior for this track
    belief_matrix[j * num_tracks + i] = posterior;
}
