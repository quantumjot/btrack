# Bayesian Tracker

**WORK IN PROGRESS**


BayesianTracker is a multi object tracking algorithm, specifically used to
reconstruct trajectories in crowded fields. Here, we use a probabilistic network
of information to perform the trajectory linking. This method uses spatial
information as well as appearance information for track linking.

The tracking algorithm assembles reliable sections of track that do not
contain splitting events (tracklets). Each new tracklet initiates a
probabilistic model, and utilises this to predict future states (and error in
states) of each of the objects in the field of view.  We assign new observations
to the growing tracklets (linking) by evaluating the posterior probability of
each potential linkage from a Bayesian belief matrix for all possible linkages.

[![SquiggleCube](http://lowe.cs.ucl.ac.uk/images/bayesian_tracker.png)](http://lowe.cs.ucl.ac.uk)
*Example of tracking objects in 3D space*

The tracklets are then assembled into tracks by using multiple hypothesis
testing and integer programming to identify a globally optimal solution. The
likelihood of each hypothesis is calculated for some or all of the tracklets
based on heuristics. The global solution identifies a sequence of
high-likelihood hypotheses that accounts for all observations.

BayesianTracker (btrack) is part of the *Sequitr* image processing toolbox for
microscopy data analysis. For more information see: http://lowe.cs.ucl.ac.uk/

---
### TODO
+ [x] Add a parameter for maximum local radius of searching
+ [ ] Finish abstract HashBin class, make the HypothesisBin and TrackBin classes
+ [ ] Complete FAST_COST_UPDATE code to minimize non-local look-ups for improved performance
+ [ ] Cross platform install scripts
+ [ ] Improve documentation
+ [ ] Migrate to Python 3

---

### Dependencies

BayesianTracker has been tested with Python 2.7+ on OS X and Linux, and requires
the following additional packages:

+ Numpy
+ Scipy
+ h5py
+ CVXOPT
+ Eigen
+ GLPK

BayesianTracker is written in C++ with a C interface to Python.

** There is now an experimental Python 3 compatible branch: btrack_py3 **

---

### Examples

We developed BayesianTracker to enable us to track individual molecules or
cells in large populations over very long periods of time, reconstruct lineages
and study cell movement or sub-cellular protein localisation. We have provided
several examples in the notebooks folder.  Below is an example of tracking
cells:

[![CellTracking](http://lowe.cs.ucl.ac.uk/images/youtube.png)](https://youtu.be/EjqluvrJGCg)

---
### Citation

More details of how the tracking algorithm works and how it can be applied to
tracking cells in time-lapse microscopy data can be found in our publication:

**Local cellular neighbourhood controls proliferation in cell competition**  
Bove A, Gradeci D, Fujita Y, Banerjee S, Charras G and Lowe AR.  
*Mol. Biol. Cell* (2017) <https://doi.org/10.1091/mbc.E17-06-0368>

```
@article{Bove07112017,
  author = {Bove, Anna and Gradeci, Daniel and Fujita, Yasuyuki and Banerjee,
    Shiladitya and Charras, Guillaume and Lowe, Alan R.},
  title = {Local cellular neighborhood controls proliferation in cell competition},
  volume = {28},
  number = {23},
  pages = {3215-3228},
  year = {2017},
  doi = {10.1091/mbc.E17-06-0368},
  URL = {http://www.molbiolcell.org/content/28/23/3215.abstract},
  eprint = {http://www.molbiolcell.org/content/28/23/3215.full.pdf+html},
  journal = {Molecular Biology of the Cell}
}
```

---

### Installation

*NOTE TO WINDOWS USERS*: We have not tested this on Windows. The setup
instructions below have been tested on Ubuntu 16.04 LTS and OS X 10.13.6.

1. First clone the BayesianTracker repo:
```sh
$ git clone https://github.com/quantumjot/BayesianTracker.git
```

2. Create a conda environment (assuming you're using Anaconda!):
```bash
$ conda create --name btrack
$ source activate btrack
$ pip install --upgrade pip
$ conda install -c conda-forge cvxopt
$ conda install h5py
$ conda install scipy
```

3. Download Eigen (https://github.com/eigenteam/eigen-git-mirror) and add this to the /btrack/include/eigen directory.

4. Finally, use the Makefile in the btrack/src to compile the code:
```sh
$ cd BayesianTracker/btrack/src
$ make
```

*NOTE*: we are working on a cross-platform install script.

---

### Usage

BayesianTracker can be used simply as follows:

```python
import btrack

# NOTE(arl):  This should be from your image segmentation code
objects = utils.import_HDF('/home/arl/Documents/objects.hdf5')

# initialise a tracker session using a context manager
with btrack.BayesianTracker() as tracker:

  # configure the tracker using a config file
  tracker.configure_from_file('cells_config.json')

  # append the objects to be tracked
  tracker.append(objects)

  # track them
  tracker.track()

  # generate hypotheses and run the global optimiser
  tracker.optimize()

  # get the tracks
  track_zero = tracker[0]
  tracks = tracker.tracks

  # export them
  # NOTE(arl): the HDF5 format stores references to the original objects
  tracker.export('/home/arl/Documents/objects.hdf5')
```

There are many additional options, including the ability to define object models.

### Input data
Observations can be provided in three basic formats:
+ a simple JSON file
+ HDF5 for larger/more complex datasets, or
+ using your own code as a PyTrackObject.

HDF5 is the *default* format for data interchange, where additional information
such as images or metadata can also be stored.  

Data can also be imported using JSON format. Here is an example of the simple JSON format:
```json
{
  "Object_203622": {
    "x": 554.29737483861709,
    "y": 1199.362071438818,
    "z": 0.0,
    "t": 862,
    "label": "interphase",
    "states": 5,
    "probability": [
      0.996992826461792,
      0.0021888131741434336,
      0.0006106126820668578,
      0.000165432647918351,
      4.232166247675195e-05
    ],
    "dummy": false
  }
}
```

### Motion models
Motion models can be written as simple JSON files which can be imported into the tracker.

```json
{
  "MotionModel": {
    "name": "Example",
    "dt": 1.0,
    "measurements": 1,
    "states": 3,
    "accuracy": 2.0,
    "A": {
      "matrix": [1,0,0]
       }
    }
}
```

Or can be built using the MotionModel class.



### Object models
To be completed.

### Hypotheses
By default, the optimizer generates 7 different hypotheses:
+ False positive (FP)
+ Initializing (init)
+ Terminating (term)
+ Link (link)
+ Branch (branch)
+ Dead (dead)
+ Merge (merge)

```json
{
  "HypothesisModel": {
    "name": "cell_hypothesis",
    "hypotheses": ["FP", "init", "term", "link", "branch", "dead"],
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
}
```
