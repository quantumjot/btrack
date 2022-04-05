#!/usr/bin/env python
# -------------------------------------------------------------------------------
# Name:     BayesianTracker
# Purpose:  A multi object tracking library, specifically used to reconstruct
#           tracks in crowded fields. Here we use a probabilistic network of
#           information to perform the trajectory linking. This method uses
#           positional and visual information for track linking.
#
# Authors:  Alan R. Lowe (arl) a.lowe@ucl.ac.uk
#
# License:  See LICENSE.md
#
# Created:  14/08/2014
# -------------------------------------------------------------------------------


__author__ = "Alan R. Lowe"
__email__ = "code@arlowe.co.uk"


import dataclasses
import os
from typing import List

import numpy as np

from . import constants, utils
from .optimise.hypothesis import H_TYPES, PyHypothesisParams


@dataclasses.dataclass
class MotionModel:
    """The `btrack` motion model.

    Parameters
    ----------
    name : str
        A name identifier for the model.
    measurements : int
        The number of measurements of the system (e.g. 3 for x, y, z).
    states : int
        The number of states of the system (typically >= measurements).
    A : array (states, states)
        State transition matrix.
    H : array (measurements, states)
        Observation matrix.
    P : array (states, states)
        Initial covariance estimate.
    G : array (1, states)
        Estimated error in process.
    R : array (measurements, measurements)
        Estimated error in measurements.
    dt : float
        Time difference (always 1)
    accuracy : float
        Integration limits for calculating the probabilities.
    max_lost : int
        Number of frames without observation before marking as lost.
    prob_not_assign : float
        The default probability to not assign a track.

    Notes
    -----
    'Is an algorithm which uses a series of measurements observed over time,
    containing noise (random variations) and other inaccuracies, and produces
    estimates of unknown variables that tend to be more precise than those that
    would be based on a single measurement alone.'

    This is just a wrapper for the data with a few convenience functions
    thrown in. Matrices must be stored Fortran style, because Eigen uses
    column major and Numpy uses row major storage.

    References
    ----------
    'A new approach to linear filtering and prediction problems.'
    Kalman RE, 1960 Journal of Basic Engineering
    """

    measurements: int
    states: int
    A: np.ndarray
    H: np.ndarray
    P: np.ndarray
    G: np.ndarray
    R: np.ndarray
    dt: float = 1.0
    accuracy: float = 2.0
    max_lost: int = constants.MAX_LOST
    prob_not_assign: float = constants.PROB_NOT_ASSIGN
    name: str = "Default"

    @property
    def Q(self):
        """Return a Q matrix from the G matrix."""
        # return self.G.transpose() * self.G
        return self.G.T @ self.G

    def reshape(self):
        """Reshapes matrices to the correct dimensions. Only need to call this
        if loading a model from a JSON file.

        Notes
        -----
        Internally:
            Eigen::Matrix<double, m, s> H;
            Eigen::Matrix<double, s, s> Q;
            Eigen::Matrix<double, s, s> P;
            Eigen::Matrix<double, m, m> R;

        """
        s = self.states
        m = self.measurements

        # if we defined a model, restructure matrices to the correct shapes
        # do some parsing to check that the model is specified correctly
        if s and m:
            shapes = {
                "A": (s, s),
                "H": (m, s),
                "P": (s, s),
                "R": (m, m),
                "G": (1, s),
            }
            for m_name in shapes:
                try:
                    m_array = getattr(self, m_name)
                    r_matrix = np.reshape(m_array, shapes[m_name], order="C")
                except ValueError:
                    raise ValueError(
                        f"Matrx {m_name} is incorrecly specified."
                        f" ({len(m_array)} entries for"
                        f" {shapes[m_name][0]}x{shapes[m_name][1]} matrix.)"
                    )

                setattr(self, m_name, r_matrix)
        else:
            raise ValueError(
                "Cannot reshape matrices as MotionModel is uninitialised."
            )

    @staticmethod
    def load(filename):
        """Load a model from file"""
        return utils.read_motion_model(filename)


@dataclasses.dataclass
class ObjectModel:
    """The `btrack` object model.

    This is a class to deal with state transitions in the object, essentially
    a Hidden Markov Model.  Makes an assumption that the states are all
    observable, but with noise.

    Parameters
    ----------
    name : str
        A name identifier for the model.
    emission : array
        The emission probability matrix.
    transition : array
        Transition probabilities.
    start : array
        Initial probabilities.
    states : int
        Number of observable states.
    """

    emission: np.ndarray
    transition: np.ndarray
    start: np.ndarray
    states: int
    name: str = "Default"

    def reshape(self):
        """Reshapes matrices to the correct dimensions. Only need to call this
        if loading a model from a JSON file.

        Notes:
            Internally:
                Eigen::Matrix<double, s, s> emission;
                Eigen::Matrix<double, s, s> transition;
                Eigen::Matrix<double, s, 1> start;
        """
        if not self.states:
            raise ValueError(
                "Cannot reshape matrices in `ObjectModel` as `states` are unknown."
            )
        s = self.states
        self.emission = np.reshape(self.emission, (s, s), order="C")
        self.transition = np.reshape(self.transition, (s, s), order="C")

    @staticmethod
    def load(filename):
        """Load a model from file"""
        return utils.read_object_model(filename)


@dataclasses.dataclass
class HypothesisModel:
    r"""The `btrack` hypothesis model.

    This is a class to deal with hypothesis generation in the optimization step
    of the tracking algorithm.

    Parameters
    ----------
    name : str
        A name identifier for the model.
    hypotheses : list[str]
        A list of hypotheses to be generated. See `optimise.hypothesis.H_TYPES`.
    lambda_time : float
        A scaling factor for the influence of time when determining
        initialization or termination hypotheses. See notes.
    lambda_dist : float
        A a scaling factor for the influence of distance at the border when
        determining initialization or termination hypotheses. See notes.
    lambda_link : float
        A scaling factor for the influence of track-to-track distance on linking
        probability. See notes.
    lambda_branch : float
        A scaling factor for the influence of cell state and position on
        division (mitosis/branching) probability. See notes.
    eta : float
        Default value for a low probability event (e.g. 1E-10) to prevent
        divide-by-zero.
    theta_dist : float
        A threshold distance from the edge of the FOV to add an initialization
        or termination hypothesis.
    theta_time : float
        A threshold time from the beginning or end of movie to add an
        initialization or termination hypothesis.
    dist_thresh : float
        Isotropic spatial bin size for considering hypotheses. Larger bin sizes
        generate more hypothesese for each tracklet.
    time_thresh : float
        Temporal bin size for considering hypotheses. Larger bin sizes generate
        more hypothesese for each tracklet.
    apop_thresh : int
        Number of apoptotic detections, counted consecutively from the back of
        the track, to be considered a real apoptosis.
    segmentation_miss_rate : float
        Miss rate for the segmentation, e.g. 1/100 segmentations incorrect gives
        a segmentation miss rate or 0.01.
    apoptosis_rate : float
        Rate of apoptosis detections.
    relax : bool
        Disables the `theta_dist` and `theta_time` thresholds when creating
        termination and initialization hypotheses. This means that tracks can
        initialize or terminate anywhere (or time) in the dataset.


    Notes
    -----
    The `lambda` (:math:`\lambda`) factors scale the probability according to
    the following function:

    .. math:: e^{(-d / \lambda)}
    """

    hypotheses: List[str]
    lambda_time: float
    lambda_dist: float
    lambda_link: float
    lambda_branch: float
    eta: float
    theta_dist: float
    theta_time: float
    dist_thresh: float
    time_thresh: float
    apop_thresh: int
    segmentation_miss_rate: float
    apoptosis_rate: float
    relax: bool
    name: str = "Default"

    @staticmethod
    def load(filename: os.PathLike):
        """Load a model from file."""
        return utils.read_hypotheis_model(filename)

    def hypotheses_to_generate(self) -> int:
        """Return an integer representation of the hypotheses to generate."""
        h_bin = "".join(
            [str(int(h)) for h in [h in self.hypotheses for h in H_TYPES]]
        )
        return int(h_bin[::-1], 2)

    def as_ctype(self) -> PyHypothesisParams:
        """Return the ctypes representation of the `HypothesisModel`."""
        h_params = PyHypothesisParams()
        fields = [f[0] for f in h_params._fields_]

        for k, v in dataclasses.asdict(self).items():
            if k in fields:
                setattr(h_params, k, v)

        # set the hypotheses to generate
        h_params.hypotheses_to_generate = self.hypotheses_to_generate()

        return h_params
