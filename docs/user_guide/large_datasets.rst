=================================
Deadling with very large datasets
=================================

btrack supports three different methods:

- ``EXACT`` - (DEFAULT) exact calculation of Bayesian belief matrix, but can be slow on large datasets
- ``APPROXIMATE`` - approximate calculation, faster, for use with large datasets.
  This has an additional ``max_search_radius`` parameter, which sets the local spatial search radius (isotropic, pixels) of the algorithm.
- ``CUDA`` - GPU implementation of the EXACT method (*in progress*)

For most cell datasets (<1000 cells per time point) we recommend ``EXACT``.
If you have larger datasets, we recommend ``APPROXIMATE``.

If the tracking does not complete, and is stuck on the optimization step, this means that your configuration is poorly suited to your data.
Try turning off optimization, followed by modifying the parameters of the config file.

.. code:: python


   import btrack
   from btrack.constants import BayesianUpdates

   with btrack.BayesianTracker() as tracker:
       # configure the tracker and change the update method
       tracker.configure_from_file('/path/to/your/models/cell_config.json')
       tracker.update_method = BayesianUpdates.APPROXIMATE
       tracker.max_search_radius = 100
       ...
