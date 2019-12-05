# Bayesian Tracker

**WORK IN PROGRESS** :microscope::computer:


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
microscopy data analysis. For more information see: http://lowe.cs.ucl.ac.uk/cellx.html

See examples of use in:  
https://github.com/quantumjot/CellTracking/tree/master/notebooks

See the wiki for more information on installation, configuration and use:
https://github.com/quantumjot/BayesianTracker/wiki

### Example: Tracking mammalian cells in time-lapse microscopy experiments

We developed BayesianTracker to enable us to track cells in large populations over very long periods of time, reconstruct lineages and study cell movement or sub-cellular protein localisation. Below is an example of tracking cells:

[![CellTracking](http://lowe.cs.ucl.ac.uk/images/youtube.png)](https://youtu.be/EjqluvrJGCg)  
*Video of tracking*

[![LineageTree](http://lowe.cs.ucl.ac.uk/images/bayesian_tracker_lineage_tree.png)](http://lowe.cs.ucl.ac.uk)  
*Automated lineage tree reconstruction*



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

You can also --> :star: :wink:

---

### Installation

BayesianTracker has been tested with Python 3.7 on OS X and Linux.
The tracker and hypothesis engine are mostly written in C++ with a C interface to Python.

*NOTE TO WINDOWS USERS*: We have not tested this on Windows. The setup
instructions below have been tested on Ubuntu 18.04 LTS and OS X 10.15.1.

1. First clone the BayesianTracker repo:
```sh
$ git clone https://github.com/quantumjot/BayesianTracker.git
```

2. (Optional, but advised) Create a conda environment:
```sh
$ conda env create -f ./BayesianTracker/environment.yml
$ conda activate btrack
```

3. Compile and install the tracker:
```sh
$ cd BayesianTracker
$ chmod +x install.sh
$ ./install.sh
```

If all goes well, you should be able to import BayesianTracker:
```python
import btrack
```

should return:
```
[INFO][2019/08/22 10:26:11 AM] btrack (v0.3.x) library imported
[INFO][2019/08/22 10:26:11 AM] Loaded btrack: <your python site-packages>/btrack/libs/libtracker.so
```



---

### Usage

BayesianTracker can be used simply as follows:

```python
import btrack
from btrack.utils import import_HDF

# NOTE(arl):  This should be from your image segmentation code
objects = import_HDF('/path/to/your/objects.hdf5')

# initialise a tracker session using a context manager
with btrack.BayesianTracker() as tracker:

  # configure the tracker using a config file
  tracker.configure_from_file('/path/to/your/models/cell_config.json')

  # append the objects to be tracked
  tracker.append(objects)

  # track them
  tracker.track()

  # generate hypotheses and run the global optimiser
  tracker.optimize()

  # get the tracks as a python list
  tracks = tracker.tracks
```

Tracks themselves are python objects with properties:

```python
# get the first track
track_zero = tracks[0]

# print the length of the track
print(len(track_zero))

# print all of the xyzt positions in the track
print(track_zero.x)
print(track_zero.y)
print(track_zero.z)
print(track_zero.t)

# print the fate of the track
print(track_zero.fate)

# print the track ID, root node, parent node and children
print(track_zero.ID)
print(track_zero.root)
print(track_zero.parent)
print(track_zero.children)

```

There are many additional options, including the ability to define object models.

### Input data
Observations can be provided in three basic formats:
+ a simple JSON file
+ HDF5 for larger/more complex datasets, or
+ using your own code as a `PyTrackObject`.

HDF5 is the *default* format for data interchange, where additional information
such as images or metadata can also be stored.  

More detail in the wiki:
https://github.com/quantumjot/BayesianTracker/wiki
