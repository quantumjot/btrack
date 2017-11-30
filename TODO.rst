TODO
====

Python
------
- Install script to run Makefile if necessary
- Tracking metrics (MOT)

C/C++ and CUDA
--------------
- Autoswitch CPU/GPU for tracking
- Update belief matrix using CUDA parallelisation
- Template track object class to use different levels of precision
- Integrate ver 0.1 track optimiser code

Misc
----
- Other motion models
- Importer/Exporters for other data analysis packages
- Change the track naming convention


Updates
=======

0.2.3
-----
- Hypothesis generation from track objects

0.2.2
-----
- HDF5 is now the default file format, for integration with conv-nets
- Tracker returns references to HDF5 groups
- Started integration of track optimiser code

0.2.1
-----
- Set limits on the volume, such that tracks which are predicted to exit the tracking volume are set to lost automatically.
- Enabled frame range in tracking to limit the range of data used
- Fast plotting of tracks
- Output a tracking statistics structure back to Python
- Track iteration to enable incremental tracking and feedback to user

0.2.0
-----
- Major update. Converted Bayesian update code to use Eigen
- Added z-dimension to tracking
