[![PyPI](https://img.shields.io/pypi/v/btrack)](https://pypi.org/project/btrack)  :microscope::computer:
<!-- [![PyPI - Downloads](https://img.shields.io/pypi/dm/btrack.svg)](https://pypistats.org/packages/btrack) -->


# Bayesian Tracker (bTrack)


http://lowe.cs.ucl.ac.uk/cellx.html


BayesianTracker (`btrack`) is a multi object tracking algorithm, specifically
used to reconstruct trajectories in crowded fields. Here, we use a
probabilistic network of information to perform the trajectory linking. This
method uses spatial information as well as appearance information for track linking.

The tracking algorithm assembles reliable sections of track that do not
contain splitting events (tracklets). Each new tracklet initiates a
probabilistic model, and utilises this to predict future states (and error in
states) of each of the objects in the field of view.  We assign new observations
to the growing tracklets (linking) by evaluating the posterior probability of
each potential linkage from a Bayesian belief matrix for all possible linkages.

<!-- [![SquiggleCube](http://lowe.cs.ucl.ac.uk/images/bayesian_tracker.png)](http://lowe.cs.ucl.ac.uk)  
*Example of tracking objects in 3D space* -->

The tracklets are then assembled into tracks by using multiple hypothesis
testing and integer programming to identify a globally optimal solution. The
likelihood of each hypothesis is calculated for some or all of the tracklets
based on heuristics. The global solution identifies a sequence of
high-likelihood hypotheses that accounts for all observations.

<!-- [![LineageTree](http://lowe.cs.ucl.ac.uk/images/bayesian_tracker_lineage_tree.png)](http://lowe.cs.ucl.ac.uk)   -->
[![LineageTree](https://raw.githubusercontent.com/quantumjot/BayesianTracker/master/examples/render.png)](http://lowe.cs.ucl.ac.uk/cellx.html)  
*Automated cell tracking and lineage tree reconstruction*. Cell divisions are highlighted in red.





### Example: Tracking mammalian cells in time-lapse microscopy experiments

We developed BayesianTracker to enable us to track cells in large populations
over very long periods of time, reconstruct lineages and study cell movement or
sub-cellular protein localisation. Below is an example of tracking cells:

[![CellTracking](http://lowe.cs.ucl.ac.uk/images/youtube.png)](https://youtu.be/EjqluvrJGCg)  
*Video of tracking*





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
The tracker and hypothesis engine are mostly written in C++ with a C interface
to Python.

*NOTE TO WINDOWS USERS*: We have not tested this on Windows, although the
following works on the Ubuntu shell for Win10. The setup instructions below have
 been tested on Ubuntu 18.04 LTS and OS X 10.15.

 #### Installing the latest stable version
 ```sh
 $ pip install btrack
 ```


 #### (Advanced) Installing the latest development version

If you would rather install the latest development version, and/or compile
directly from source, you can clone and install from this repo:

```sh
$ git clone https://github.com/quantumjot/BayesianTracker.git
$ conda env create -f ./BayesianTracker/environment.yml
$ conda activate btrack
$ cd BayesianTracker
$ pip install -e .
```

Addtionally, the `build.sh` script will download Eigen source, run the makefile
and pip install.

---
### Usage in Colab notebooks

If you do not want to install a local copy, you can run the tracker in a Colab notebook. Please note that these examples are work in progress and may change:

| Status        | Level | Notebook                                     | Link |
| ------------- | ----- | -------------------------------------------- | ---- |
| *In progress* | Basic | Loading your own data                        | -
| Complete      | Basic | Object tracking with btrack                  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A1PRM0a3Z0ufszdnVxntcaEDzU_Vh4u9)|
| Complete      | Basic | Data import options                          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V2TtJ5FGqSILTuThSRg5j9crsBsorUmy)|
| *In progress* | Advanced | Object tracking with btrack (2D/3D)       | -
| *In progress* | Advanced | Configuration options                     | -
| Complete      | Advanced | How to compile btrack from source         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19t5HBV76_Js8M3LX63CwiXzemax7Tvsk)|


---

### Usage with Napari

You can visualize the output using our plugin (called `arboretum`) for the open source image viewer [`Napari`](https://github.com/napari/napari). We provide example code here:

| Status        | Notebook                                     | Link |
| ------------- | -------------------------------------------- | ---- |
| *In progress* | Visualizing bTrack output using arboretum    | [GitHub](https://github.com/quantumjot/arboretum)

---

### Usage from Python

BayesianTracker can be used simply as follows:

```python
import btrack
from btrack.utils import import_HDF

# NOTE(arl):  This should be from your image segmentation code
objects = import_HDF('/path/to/your/objects.hdf5', filter_using=None)

# initialise a tracker session using a context manager
with btrack.BayesianTracker() as tracker:

  # configure the tracker using a config file
  tracker.configure_from_file('/path/to/your/models/cell_config.json')

  # append the objects to be tracked
  tracker.append(objects)

  # set the volume (Z axis volume is set very large for 2D data)
  tracker.volume=((0,1200),(0,1600),(-1e5,1e5))

  # track them (in interactive mode)
  tracker.track_interactive(step_size=100)

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

Tracks can also be exported in the LBEP format:
```python
from btrack.utils import export_LBEP

export_LBEP('/path/to/your/res_track.txt', tracks)
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
https://github.com/quantumjot/BayesianTracker/wiki/3.-Importing-data
