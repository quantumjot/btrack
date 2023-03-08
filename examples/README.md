## BayesianTracker examples

Example notebooks can be found in this directory.

Example datasets and configurations can be found at:
https://github.com/lowe-lab-ucl/btrack-examples

Or, alternatively can be accessed using `btrack` itself:

```python

from btrack import datasets

# example segmentation
segmentation = datasets.example_segmentation()

# example config
config = datasets.cell_config()

# example objects
track_objects = datasets.example_track_objects()
```
