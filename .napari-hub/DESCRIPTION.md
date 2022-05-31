[![PyPI](https://img.shields.io/pypi/v/btrack)](https://pypi.org/project/btrack)
[![Downloads](https://pepy.tech/badge/btrack/month)](https://pepy.tech/project/btrack)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/quantumjot/BayesianTracker/actions/workflows/test.yml/badge.svg)](https://github.com/quantumjot/BayesianTracker/actions/workflows/test.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Documentation](https://readthedocs.org/projects/btrack/badge/?version=latest)](https://btrack.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/quantumjot/BayesianTracker/branch/main/graph/badge.svg?token=QCFC9AWK0R)](https://codecov.io/gh/quantumjot/BayesianTracker)
![logo](https://btrack.readthedocs.io/en/latest/_images/btrack_logo.png)


BayesianTracker (`btrack`) is a multi object tracking algorithm,
specifically used to reconstruct trajectories in crowded fields.  New
observations are assigned to tracks by evaluating the posterior probability of
each potential linkage from a Bayesian belief matrix for all possible
linkages.

We developed `btrack` for cell tracking in time-lapse microscopy data.
