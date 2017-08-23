# Bayesian Tracker

** WORK IN PROGRESS ** (Last update: 23/08/2017)


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


[![SquiggleBox](http://lowe.cs.ucl.ac.uk/images/tracks.png)]()
*Example of tracking objects in 3D space*

BayesianTracker (btrack) is part of the ImPy image processing toolbox for
microscopy data analysis. For more information see: http://lowe.cs.ucl.ac.uk/

---

### Requirements

BayesianTracker has been tested with Python 2.7+ on OS X and Linux, and requires
the following additional packages:

+ Numpy
+ Scipy
+ Matplotlib (Optional)
+ Eigen (Optional)
+ Jupyter (Optional)

---

### Examples

We developed BayesianTracker to enable us to track individual molecules or
cells in large populations over very long periods of time, reconstruct lineages
and study cell movement or sub-cellular protein localisation. Below is an
example of tracking cells:

[![CellTracking](http://lowe.cs.ucl.ac.uk/images/youtube.png)](https://youtu.be/dsjUnRwu33k)

---

### Installation

You can install BayesianTracker by cloning the repo and running the setup script:
```sh
$ git clone https://github.com/quantumjot/BayesianTracker.git
$ cd BayesianTracker
$ python setup.py install
```



---

### Principles of operation
To be completed.

### Usage

BayesianTracker can be used simply as follows:

```python
import btrack

# NOTE:  This should be from your image segmentation code
objects = [btrack.TrackObject(t) for t in observations]

# initialise a tracker session using a context manager
with btrack.BayesianTracker() as tracker:
  # append an object to be tracked
  for objs in objects:
    tracker.append(obj)

  # track them
  tracker.track()

  # get the tracks
  track_zero = tracker[0]
  tracks = tracker.tracks

  # export them
  tracker.export('/home/arl/Documents/tracks.json')
```

There are many additional options, including the ability to define object models.
More details will be provided.

### References
