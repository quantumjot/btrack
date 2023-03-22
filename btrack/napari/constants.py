"""
This module contains variables that are used throughout the
napari_btrack package.
"""

HYPOTHESES = [
    "P_FP",
    "P_init",
    "P_term",
    "P_link",
    "P_branch",
    "P_dead",
    "P_merge",
]

HYPOTHESIS_SCALING_FACTORS = [
    "lambda_time",
    "lambda_dist",
    "lambda_link",
    "lambda_branch",
]

HYPOTHESIS_THRESHOLDS = [
    "theta_dist",
    "theta_time",
    "dist_thresh",
    "time_thresh",
    "apop_thresh",
    "relax",
]
