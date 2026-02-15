# Bayesian Algorithm Analysis for btrack

## Executive Summary

I've analyzed the core Bayesian update algorithm in the C++ implementation. I found **two critical errors** and several opportunities for improvement. The most serious issue is in the Hidden Markov Model forward algorithm (inference.cc), which has a fundamental implementation error that would prevent it from working correctly.

---

## Critical Issues

### 1. **CRITICAL: Forward Algorithm Implementation Error**
**Location**: `btrack/src/inference.cc:38-44`

**Current Implementation**:
```cpp
void ObjectModel::forward(const unsigned int observation) {
  Eigen::VectorXd update(states);
  for (unsigned int s = 0; s < states; s++) {
    update = transition.col(s).array() * x_hat.array();
    x_hat(s) = emission(s, observation) * update.sum();
  }
}
```

**Problem**: The `update` vector is being **overwritten in each loop iteration** and then used to update `x_hat(s)` directly. This violates the fundamental principle of the forward algorithm, where the new state should depend on ALL previous states weighted by transition probabilities, not just computed iteratively in-place.

**Impact**: This causes the algorithm to:
- Overwrite intermediate results
- Use partially updated state values to compute subsequent states
- Produce incorrect probability estimates

**Correct Implementation**:
```cpp
void ObjectModel::forward(const unsigned int observation) {
  Eigen::VectorXd x_hat_new(states);
  for (unsigned int s = 0; s < states; s++) {
    // Sum over all previous states weighted by transition probability
    x_hat_new(s) = emission(s, observation) *
                   (transition.row(s).array() * x_hat.array()).sum();
  }
  x_hat = x_hat_new;  // Update only after computing all new states
}
```

**Even Better** (vectorized):
```cpp
void ObjectModel::forward(const unsigned int observation) {
  // x_new = emission(obs) .* (Transition^T * x_old)
  x_hat = (transition.transpose() * x_hat).array() *
          emission.col(observation).array();

  // Normalize to prevent numerical underflow
  x_hat /= x_hat.sum();
}
```

---

### 2. **POTENTIAL ERROR: Bayesian Update Formula**
**Location**: `btrack/src/bayes.cc:26`

**Current Implementation**:
```cpp
double PrDP = prob_assign * safe_prior_assign +
              prob_not_assign * (1. - prob_assign);
double posterior = (prob_assign * (safe_prior_assign / PrDP));
```

**Problem**: The marginal probability calculation looks incorrect. The formula computes:
```
P(D) = P(D|H) * P(H) + P(D|¬H) * (1 - P(D|H))
```

But the correct formula for the marginal probability should be:
```
P(D) = P(D|H) * P(H) + P(D|¬H) * P(¬H)
```

**Current**: `prob_not_assign * (1. - prob_assign)`
**Should be**: `prob_not_assign * (1. - safe_prior_assign)`

**Corrected Code**:
```cpp
double inv_prior_assign = 1.0 - safe_prior_assign;
double PrDP = prob_assign * safe_prior_assign +
              prob_not_assign * inv_prior_assign;
double posterior = (prob_assign * safe_prior_assign) / PrDP;
```

**Analysis**: The current code uses `(1. - prob_assign)` which represents the complement of the likelihood, not the prior. This would cause incorrect posterior probabilities, especially when `prob_assign` is high but the prior is low, or vice versa.

---

## Moderate Issues

### 3. Backward Algorithm Not Implemented
**Location**: `btrack/src/inference.cc:47`

```cpp
void ObjectModel::backward() {}
```

**Issue**: The backward algorithm is completely empty, meaning the Hidden Markov Model only performs forward filtering, not forward-backward smoothing.

**Impact**:
- Cannot perform optimal state estimation using both past and future observations
- Cannot compute smoothed probabilities P(state | all observations)
- Limited to filtering (P(state | past observations))

**Recommendation**: Either implement the backward algorithm or remove the unused method and associated functionality.

---

### 4. Missing Normalization in Bayesian Updates
**Location**: `btrack/src/tracker.cc:479-533` (cost_EXACT function)

**Issue**: After performing Bayesian updates across all objects, there's no explicit normalization of the belief matrix. While the Bayesian update theoretically maintains normalization, numerical errors can accumulate.

**Recommendation**: Add explicit normalization after all updates:
```cpp
// After line 527
belief.col(trk) = v_posterior;

// Add normalization
double sum = v_posterior.sum();
if (sum > 1e-10) {  // Avoid division by zero
  belief.col(trk) = v_posterior / sum;
}
```

---

### 5. Numerical Stability Concerns

**Location**: `btrack/src/bayes.cc:23`

**Current Code**:
```cpp
double safe_prior_assign = std::max(prior_assign, 1e-99);
```

**Issue**: Using `1e-99` as a minimum value can lead to numerical instability when computing ratios and products. This is well below the double precision epsilon (~1e-308 for denormalized, ~2.2e-16 for relative precision).

**Recommendation**: Use a more conservative threshold:
```cpp
double safe_prior_assign = std::max(prior_assign, 1e-10);
```

---

## Minor Issues and Improvements

### 6. Multivariate Error Function Approximation
**Location**: `btrack/src/pdf.cc:46-71`

**Current Approach**: Treats each spatial dimension independently and multiplies probabilities:
```cpp
phi *= .5 * phi_x;
```

**Note**: This is an approximation that assumes diagonal covariance (independent axes). While this is documented in comments and probably acceptable for the use case, it's not a true multivariate Gaussian probability.

**Recommendation**: Document this assumption more clearly in the function header, or consider implementing the full multivariate normal CDF if accuracy is critical.

---

### 7. Inefficient Update Vector Operations
**Location**: `btrack/src/tracker.cc:516-522`

**Current Code**:
```cpp
v_update.fill(safe_update);
v_update(obj) = 1.;
v_posterior = v_posterior.array() * v_update.array();
v_posterior(obj) = posterior;
```

**Issue**: This creates and fills an update vector for each object, even though we're only updating one element differently.

**More Efficient**:
```cpp
// Update all except current object
v_posterior.array() *= safe_update;
// Set current object's posterior
v_posterior(obj) = posterior;
```

---

### 8. Cosine Similarity Edge Case
**Location**: `btrack/src/pdf.cc:81-82`

**Current Code**:
```cpp
double f_mag = (trk_last->features).norm() * (obj->features).norm();
double cosine_similarity = f_dot / f_mag;
```

**Issue**: If either feature vector has zero norm, this will produce NaN or Inf.

**Recommendation**: Add explicit zero-norm check:
```cpp
double f_mag = (trk_last->features).norm() * (obj->features).norm();
if (f_mag < 1e-10) {
  return 0.5;  // Default neutral similarity
}
double cosine_similarity = f_dot / f_mag;
```

---

## Testing Recommendations

To verify these issues and validate fixes, I recommend:

1. **Unit Tests for Forward Algorithm**:
   - Test with known HMM transition/emission matrices
   - Compare against reference implementations (e.g., hmmlearn in Python)
   - Verify probability conservation (sum to 1)

2. **Unit Tests for Bayesian Updates**:
   - Test with extreme prior values (0.01, 0.99)
   - Test with extreme likelihood values
   - Verify posterior probabilities sum to 1
   - Compare against manual Bayes rule calculations

3. **Integration Tests**:
   - Create synthetic tracking data with known ground truth
   - Measure tracking accuracy before and after fixes
   - Profile numerical stability with long tracks

4. **Numerical Stability Tests**:
   - Test with very small probabilities
   - Test with very long tracks (>1000 frames)
   - Monitor for NaN/Inf values in belief matrices

---

## Priority Recommendations

### Immediate (Critical):
1. **Fix the forward algorithm** in `inference.cc` - this is a fundamental error
2. **Verify and fix the Bayesian update formula** in `bayes.cc:26`

### High Priority:
3. Add explicit normalization in belief matrix updates
4. Improve numerical stability thresholds
5. Add zero-norm checks in cosine similarity

### Medium Priority:
6. Decide on backward algorithm: implement or remove
7. Optimize update vector operations
8. Add comprehensive unit tests for all probability calculations

---

## Additional Notes

### Strengths of Current Implementation:
- Good separation of concerns (motion model, object model, tracking)
- Use of Eigen for efficient matrix operations
- Safety checks for NaN values in several places
- Configurable update modes (EXACT vs APPROXIMATE)

### Architecture Observations:
- The overall Bayesian tracking approach is sound
- The Kalman filter implementation for motion modeling appears correct
- The hypothesis generation and linking logic is well-structured
- The use of belief matrices for multi-object tracking is appropriate

---

## Validation

To validate this analysis, you can:

1. Check if the ObjectModel is actually being used in tracking (search for its usage in tracker.cc)
2. Create a simple test case with known probabilities to verify the Bayesian update
3. Run the existing test suite and check for any failures related to probability calculations
4. Add logging to output belief matrix values and verify they remain normalized

Would you like me to:
- Create a script to test these issues?
- Implement the fixes?
- Create unit tests for the probability calculations?
