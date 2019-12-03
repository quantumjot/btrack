TODO
====

Python
------
- Tracking metrics (MOT)
- Properly pass softmax output to C++

C/C++ and CUDA
--------------
- Return lineage trees to python
- Add merge hypothesis
- Prediction class defaults to six states, make this model agnostic
- Update belief matrix using CUDA parallelisation
- Give each track a unique hash to make sure we don't overwrite IDs
- Test other motion model instantiation (sp. Prediction defaults to 6 states)
- Use softmax score to weight hypothesis generation

Misc
----
- Other motion models
- Importer/Exporters for other data analysis packages
- Change the track naming convention



Updates
=======

0.3.0
-----
- Migrated to Python 3.7
- Updated (cleaned) Makefile for easier installation
- Simplified hypothesis generation
- Added extra feedback to user including different hypotheses for intialization and termination of tracks
- Tracks can be appended to HDF input files
- Bug fixes to MATLAB exporter
- Fixed bug with sample config and when returning intermediate output of Kalman filter

0.2.13
------
- Added lineage tree creation to TrackManager in C++ lib
- Added track lineage tree creation
- Simplified code for conversion to Python 3

0.2.12
------
- Improved HDF reader for performance
- Added back ability to write tracks to HDF file
- Changed tracklet base type to refer to pointers to original objects
- Better handling of object metadata with new Tracklet class
- Added enumetated States, Fates and Errors to python lib

0.2.11
------
- Cleaned repo structure for deployment
- Add a git clone of eigen during installation (if required)
- Tested install and compilation scripts on linux
- Changed model loading to user defined directory

0.2.10
------
- Simplified model configuration format
- Allowed a second user model directory, to supplement core models
- Added install scripts

0.2.9
-----
- Changed default apoptosis hypothesis calculation to reflect relative number of observations
- Removed lineage tree generation (now part of Sequitr)
- Improved JSON export

0.2.8
-----
- Added children to return type
- Uses internal track ID for reference
- Added a split track function, using a rule to split
- Add windows compatible __declspec(dllexport) for .DLL compilation (not tested)
- Added set_volume function to define the imaging volume

0.2.7
-----
- Moved btrack types to seperate lib to help migration to python 3
- Added a fast update option that only evaluates local trajectories fully

0.2.6
-----
- Added get_motion_vector function to motion model to make predictions more
  model agnostic
- Added the ability to select which hypotheses are generated during optimization
- Added more tracking statistics to logging
- Improved track linking heuristics
- Minor bug fixes to log likelihood calculations

0.2.5
-----
- Changed default logger to work with Sequitr GPU server
- Cleaned up rendering of tracks for Jupyter notebooks
- Added time dimension to 'volume' cropping
- Added fate property to tracks

0.2.4
-----
- Returns dummy objects to HDF5 writer
- Returns parent ID from tracks to enable lineage tree creation

0.2.3
-----
- Hypothesis generation from track objects, integration of new Eigen code
- Hypothesis based track optimisation using GLPK
- Track merging moved to C++ code as part of track manager

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
