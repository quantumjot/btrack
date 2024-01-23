[![PyPI](https://img.shields.io/pypi/v/btrack)](https://pypi.org/project/btrack)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/btrack.svg)](https://python.org)
[![Downloads](https://pepy.tech/badge/btrack/month)](https://pepy.tech/project/btrack)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/quantumjot/btrack/actions/workflows/test.yml/badge.svg)](https://github.com/quantumjot/btrack/actions/workflows/test.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Documentation](https://readthedocs.org/projects/btrack/badge/?version=latest)](https://btrack.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/quantumjot/btrack/branch/main/graph/badge.svg?token=QCFC9AWK0R)](https://codecov.io/gh/quantumjot/btrack)
[![doi:10.3389/fcomp.2021.734559](https://img.shields.io/badge/doi-10.3389%2Ffcomp.2021.734559-blue)](https://doi.org/10.3389/fcomp.2021.734559)

![logo](https://btrack.readthedocs.io/en/latest/_images/btrack_logo.png)


`btrack` is a multi object tracking algorithm,
specifically used to reconstruct trajectories in crowded fields.  New
observations are assigned to tracks by evaluating the posterior probability of
each potential linkage from a Bayesian belief matrix for all possible
linkages.

We developed `btrack` for cell tracking in time-lapse microscopy data.

![tracking2](https://github.com/quantumjot/btrack/assets/8217795/7b16381a-b7e5-4750-98c6-bcdfbe95b908)

<!--
## tutorials

* https://napari.org/tutorials/tracking/cell_tracking.html
-->

## Installation

To install the `napari` plugin associated with `btrack` run the command.

```sh
pip install btrack[napari]
```

## Example data

You can try out the btrack plugin using sample data:

```sh
python btrack/napari/examples/show_btrack_widget.py
```

which will launch `napari` and the `btrack` widget, along with some sample data.


## Setting parameters

There are detailed tips and instructions on parameter settings over at the [documentation](https://btrack.readthedocs.io/en/latest/user_guide/index.html).


## Associated plugins

* [napari-arboretum](https://www.napari-hub.org/plugins/napari-arboretum) - Napari plugin to enable track graph and lineage tree visualization.
