TODO
====

Python
------

C/C++ and CUDA
--------------
- Add merge hypothesis
- Prediction class defaults to six states, make this model agnostic
- Update belief matrix using CUDA parallelisation
- Give each track a unique hash to make sure we don't overwrite IDs
- Test other motion model instantiation (sp. Prediction defaults to 6 states)
- Use softmax score to weight hypothesis generation

Misc
----
- Other motion models
- Add documentation/wiki for configuration



Thanks
======
- Thanks to Giulia V and Kristina U for helpful suggestions.
- Thanks to Volker H for notebook extensions



Updates
=======
0.4.2
-----
- Bug fix for object properties. These are now updated rather than being overwritten
- Bug fix for writing multiple object types to the same h5 file
- Added project logo
- Major update of docstrings to follow numpy guidelines
- Added a separate configuration guide document to the repo

0.4.1
-----
- Added track and object properties which store associated metadata
- Added more tests
- Added a segmentation to objects utility function
- Allow properties to be determined directly using scikit-image
- HDF5FileHandler can now write objects directly

0.4.0
------
- Use long C type for object ID
- Refactored HDF5FileHandler for improved loading/saving
- Added `refs` property to `Tracklets` to enable indexing to original objects
- Deprecated `_PyTrackObjectFactory` for generating objects
- Added `objects_from_dict` and `objects_from_array` convenience functions
- Reformatted code with black/flake8
- Added .github/workflow

0.3.13
------
- Added convenience function to convert Pandas or numpy arrays to PyTrackObject
- Added __repr__ for object and tracklet representations in jupyter notebooks
- Deprecated import_HDF in favor of the HDF5FileHandler
- Bug fix for CSV importer
- Added tracker.to_napari() for integration with napari track layer
- Added change to logging verbosity when instantiating tracker


0.3.12
------
- Added radial clipping option to linker
- Exposed solver options to tracker interface

0.3.11
------
- Added Windows support

0.3.10
------
- CSV object importer
- CSV track exporter
- Removed JSON export
- Added children to tracks when loading from hdf
- Added checks to hdf handler to make sure properties exist when loading/saving
- Added is_leaf, start and stop properties to tracklet class

0.3.9
-----
- Added approximate Bayesian updates for use with very large datasets/numbers of objects. Reduces compute time at expense of completeness.
- Added generational depth to the tree output
- Simplify IO/cleaned up HDF handler

0.3.8
-----
- Added 'lazy' termination and initialization hypotheses to aid diagnostics
- Store library version number in HDF files
- Fixed error where last frame of data not added to tracks
- Removed python lineage tree generation (now performed in C++ lib)
- Remove JSON track loader/exporter

0.3.7
-----
- Root nodes now how correct root and parent ID set by track manager
- Makefile infers version number from VERSION.txt during build from source
- Added _build_track_from_dict to JSON loader
- Fixed error writing filtering to HDF

0.3.6
-----
- PyPi release
- Added property filtering to export_delegator, and HDF exporter

0.3.5
-----
- Sanity check to ensure shared library has same version as python wrapper
- Filtering options stored in HDF file now
- Preparations for pip registration

0.3.4
-----
- Store library version in shared lib
- Preparations for pip package

0.3.3
-----
- Improved HDF loader to allow recovery of tracks and trees into native format
- Fixed JSON loader
- Added example tracking data to the repository
- Bug fix to ObjectFactory
- Provided better sample configuration file

0.3.2
-----
- Added a generic filtering option when retrieving objects from HDF files
- Fixes some small bugs and updated documentation

0.3.1
-----
- Added new states {NULL, DUMMY} to PyTrackObject
- Changed default dummy insertion behavior, now given a DUMMY state (removed)
- Added VERSION.txt for quick update of version numbers
- Small update to Python packaging
- Cleaned repo structure, moved load_config to utils
- Fixed bug with default class labels in ObjectFactory
- Added an LBEP exporter for future integration with Napari?
- Unified data export methods, now use tracker.export()

0.3.0
-----
- Migrated to Python 3.7
- Updated (cleaned) Makefile for easier installation
- Simplified hypothesis generation
- Added extra feedback to user including different hypotheses for initialization and termination of tracks
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
- Added get_motion_vector function to motion model to make predictions more model agnostic
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
