Guide to ``btrack`` configuration parameters
============================================

This is a short guide to the configuration parameters for ``btrack``

Miscellaneous parameters
------------------------

- ``max_search_radius`` - the maximum search radius in isotropic unit of the data
- ``mode`` - the update mode for the tracker
- ``volume`` - estimate of the dimensions of the imaging volume

Motion model
------------

.. code:: json

   "MotionModel": {
     "name": "cell_motion",
     "dt": 1.0,
     "measurements": 3,
     "states": 6,
     "accuracy": 7.5,
     "prob_not_assign": 0.001,
     "max_lost": 5,
     "A": {
       "matrix": [1,0,0,1,0,0,
                  0,1,0,0,1,0,
                  0,0,1,0,0,1,
                  0,0,0,1,0,0,
                  0,0,0,0,1,0,
                  0,0,0,0,0,1]
     },
     "H": {
       "matrix": [1,0,0,0,0,0,
                  0,1,0,0,0,0,
                  0,0,1,0,0,0]
     },
     "P": {
       "sigma": 150.0,
       "matrix": [0.1,0,0,0,0,0,
                  0,0.1,0,0,0,0,
                  0,0,0.1,0,0,0,
                  0,0,0,1,0,0,
                  0,0,0,0,1,0,
                  0,0,0,0,0,1]
     },
     "G": {
       "sigma": 15.0,
       "matrix": [0.5,0.5,0.5,1,1,1]

     },
     "R": {
       "sigma": 5.0,
       "matrix": [1,0,0,
                  0,1,0,
                  0,0,1]
     }
   }

- ``name`` - this is the name of the model
- ``measurements`` - the number of measurements of the system (e.g. x, y, z)
- ``states`` - the number of states of the system (typically >= measurements)
- ``A`` - State transition matrix
- ``B`` - Control matrix
- ``H`` - Observation matrix
- ``P`` - Initial covariance estimate
- ``Q`` - Estimated error in process
- ``R`` - Estimated error in measurements
- ``accuracy`` - integration limits for calculating the probabilities
- ``dt`` - time difference, always 1
- ``max_lost`` - number of frames without observation before marking a track as lost
- ``prob_not_assign`` - the default probability to not assign a track
- ``sigma`` - a scalar multiplication factor used for each matrix

Hypothesis model
----------------

Below is an example of a configuration for the global optimizer.

.. code:: json

   "HypothesisModel": {
     "name": "cell_hypothesis",
     "hypotheses": ["P_FP", "P_init", "P_term", "P_link", "P_branch", "P_dead"],
     "lambda_time": 5.0,
     "lambda_dist": 5.0,
     "lambda_link": 5.0,
     "lambda_branch": 5.0,
     "eta": 1e-150,
     "theta_dist": 5.0,
     "theta_time": 5.0,
     "dist_thresh": 10,
     "time_thresh": 3,
     "apop_thresh": 2,
     "segmentation_miss_rate": 0.1,
     "apoptosis_rate": 0.1,
     "relax": false
   }

The parameters are, as follows:

- ``name`` - this is the name of the model
- ``hypotheses`` - this is a list of hypotheses to generate for the optimizer
- ``lambda_time`` - a scaling factor for the influence of time when determining initialization or termination hypotheses
- ``lambda_dist`` - a scaling factor for the influence of distance at the border when determining initialization or termination hypotheses
- ``lambda_link`` - a scaling factor for the influence of track-to-track distance on linking probability
- ``lambda_branch`` - a scaling factor for the influence of cell state and position on division (mitosis/branching) probability
- ``eta`` - default low probability
- ``theta_dist`` - a threshold (in pixels) for the distance from the edge of the FOV to add an initialization or termination hypothesis
- ``theta_time`` - a threshold (in frames) for the time from the beginning or end of movie to add an initialization or termination hypothesis
- ``dist_thresh`` - bin size for considering hypotheses
- ``time_thresh`` - bin size for considering hypotheses
- ``apop_thresh`` - number of apoptotic detections, counted consecutively from the back of the track, to be considered a real apoptosis
- ``segmentation_miss_rate`` - miss rate for the segmentation, e.g. 1/100 segmentations incorrect = 0.01
- ``apoptosis_rate`` - rate of apoptosis detections
- ``relax`` - disables the ``theta_dist`` and ``theta_time`` thresholds to create termination and intialization hypotheses

Object model
------------
