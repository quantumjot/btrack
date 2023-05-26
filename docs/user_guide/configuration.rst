Guide to ``btrack`` configuration parameters
============================================

This is a short guide to the configuration parameters for ``btrack``.

.. note::
  Example configurations for particle or cell tracking applications can be found in the `models/` folder.


Motion model
------------

The motion model is used to make forward predictions about the location of objects using historical information and estimates of the error in the measurements and the process.

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

.. note::
  In particular, the values of `sigma` for the matrices `P`, `G` and `R` specify the magnitude of the error in the initial estimates, the process itself and the measurements.

Detailed descriptions of the other parameters of the model can be found in the API documentation:

* :py:class:`btrack.models.MotionModel`

Hypothesis model
----------------

The hypothesis model is used by the global optimizer to build the final set of tracks if using the :py:meth:`btrack.BayesianTracker.optimize()` method.

.. code:: json

   "HypothesisModel": {
     "name": "cell_hypothesis",
     "hypotheses": ["P_FP", "P_init", "P_term", "P_link", "P_branch", "P_dead"],
     "lambda_time": 5.0,
     "lambda_dist": 5.0,
     "lambda_link": 5.0,
     "lambda_branch": 5.0,
     "eta": 1e-10,
     "theta_dist": 5.0,
     "theta_time": 5.0,
     "dist_thresh": 10,
     "time_thresh": 3,
     "apop_thresh": 2,
     "segmentation_miss_rate": 0.1,
     "apoptosis_rate": 0.1,
     "relax": false
   }

.. note::
  The `hypotheses` field contains a list of hypotheses to generate while running the global optimizer. The hypotheses can be chosen from the following options:

  * `P_FP` - Hypothesis that a tracklet is a false positive detection.
  * `P_init` - Hypothesis that a tracklet starts at the beginning of the movie or edge of the FOV.
  * `P_term` - Hypothesis that a tracklet ends at the end of the movie or edge of the FOV.
  * `P_link` - Hypothesis that two tracklets should be linked together.
  * `P_branch` - Hypothesis that a tracklet can split onto two daughter tracklets.
  * `P_dead` - Hypothesis that a tracklet terminates without leaving the FOV.
  * `P_merge` - Hypothesis that two tracklets merge into one tracklet.

  The list must contain at least `P_FP`.

Detailed descriptions of the other parameters of the model can be found in the API documentation:

* :py:class:`btrack.models.HypothesisModel`

Miscellaneous parameters
------------------------

General tracking configuration options are detailed in :py:class:`btrack.config.TrackerConfig`.

- ``max_search_radius`` - The maximum search radius for the tracking algorithm in isotropic unit of the data. This parameter can be used to prevent very large displacements when linking objects.
- ``update_mode`` - The update mode for the tracker. The default option considers all possible combinations of linking objects, so can be slow for very large datasets. See :ref:`update_methods` for more information.
- ``volume`` - An estimate of the dimensions of the imaging volume, used to define the edges of the field of view for generating hypotheses and labeling tracks as lost.

Tips and warnings
-----------------

.. warning::
  The output of the tracking is very sensitive to the choice of parameter values. We suggest that you first optimize the motion model parameters, without using the optimization step (i.e. do not use :py:meth:`btrack.BayesianTracker.optimize()` initially).  Once you are satisfied with the intermediate results, proceed to optimizing the hypothesis model parameters.

.. warning::
  The global optimization step can take a very long time to complete if you have a poor choice of model parameters. By default, the optimizer will time-out after 60 seconds of attempting to solve to optimization.
