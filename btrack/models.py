from typing import Any, Optional

import numpy as np
from numpy import typing as npt
from pydantic import BaseModel, field_validator, model_validator
from pydantic_core import core_schema

from . import constants
from .optimise.hypothesis import H_TYPES, PyHypothesisParams

__all__ = ["MotionModel", "ObjectModel", "HypothesisModel"]


def _check_symmetric(x: npt.NDArray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Check that a matrix is symmetric by comparing with it's own transpose."""
    return np.allclose(x, x.T, rtol=rtol, atol=atol)


class MotionModel(BaseModel):
    r"""The `btrack` motion model.

    Parameters
    ----------
    name : str
        A name identifier for the model.
    measurements : int
        The number of measurements of the system (e.g. 3 for x, y, z).
    states : int
        The number of states of the system (typically >= measurements). The
        standard states for a constant velocity model are (x, y, z, dx, dy, dz),
        i.e. 6 in total for 3 measurements.
    A : array (states, states)
        State transition matrix.
    H : array (measurements, states)
        Observation matrix.
    P : array (states, states)
        Initial covariance estimate.
    G : array (1, states), optional
        Simplified estimated error in process. Is used to calculate Q using
        Q = G.T @ G. Either G or Q must be defined.
    Q : array (states, states), optional
        Full estimated error in process. Either G or Q must be defined.
    R : array (measurements, measurements)
        Estimated error in measurements.
    dt : float
        Time difference (always 1).
    accuracy : float
        Integration limits for calculating the probabilities.
    max_lost : int
        Number of frames without observation before marking as lost.
    prob_not_assign : float
        The default probability to not assign a track.

    Notes
    -----
    Uses a Kalman filter [1]_:

    'Is an algorithm which uses a series of measurements observed over time,
    containing noise (random variations) and other inaccuracies, and produces
    estimates of unknown variables that tend to be more precise than those that
    would be based on a single measurement alone.'

    Predicted estimate of state:

    .. math:: \hat{x}_{t\vert~t-1} = A_t \hat{x}_{t-1\vert~t-1}

    Predicted estimate of covariance:

    .. math:: P_{t\vert~t-1} = A_t P_{t-1\vert~t-1} A_t^{\top} + Q_t

    This is just a wrapper for the data with a few convenience functions
    thrown in. Matrices must be stored Fortran style, because Eigen uses
    column major and Numpy uses row major storage.

    References
    ----------
    .. [1] 'A new approach to linear filtering and prediction problems.'
      Kalman RE, 1960 Journal of Basic Engineering
    """

    measurements: int
    states: int
    A: npt.NDArray
    H: npt.NDArray
    P: npt.NDArray
    R: npt.NDArray
    G: Optional[npt.NDArray] = None
    Q: Optional[npt.NDArray] = None
    dt: float = 1.0
    accuracy: float = 2.0
    max_lost: int = constants.MAX_LOST
    prob_not_assign: float = constants.PROB_NOT_ASSIGN
    name: str = "Default"

    @field_validator("A", "H", "P", "R", "G", "Q", mode="before")
    @classmethod
    def parse_arrays(cls, v):
        if isinstance(v, dict):
            m = v.get("matrix", None)
            s = v.get("sigma", 1.0)
            return np.asarray(m, dtype=float) * s
        return np.asarray(v, dtype=float)

    @field_validator("A")
    @classmethod
    def reshape_A(cls, a, info):
        values = info.data
        shape = (values["states"], values["states"])
        return np.reshape(a, shape)

    @field_validator("H")
    @classmethod
    def reshape_H(cls, h, info):
        values = info.data
        shape = (values["measurements"], values["states"])
        return np.reshape(h, shape)

    @field_validator("P")
    @classmethod
    def reshape_P(cls, p, info):
        values = info.data
        shape = (values["states"], values["states"])
        p = np.reshape(p, shape)
        if not _check_symmetric(p):
            raise ValueError("Matrix `P` is not symmetric.")
        return p

    @field_validator("R")
    @classmethod
    def reshape_R(cls, r, info):
        values = info.data
        shape = (values["measurements"], values["measurements"])
        r = np.reshape(r, shape)
        if not _check_symmetric(r):
            raise ValueError("Matrix `R` is not symmetric.")
        return r

    @field_validator("G")
    @classmethod
    def reshape_G(cls, g, info):
        values = info.data
        shape = (1, values["states"])
        return np.reshape(g, shape)

    @field_validator("Q")
    @classmethod
    def reshape_Q(cls, q, info):
        values = info.data
        shape = (values["states"], values["states"])
        q = np.reshape(q, shape)
        if not _check_symmetric(q):
            raise ValueError("Matrix `Q` is not symmetric.")
        return q

    @model_validator(mode="after")
    def validate_motion_model(self):
        if self.Q is None:
            if self.G is None:
                raise ValueError("Either a `G` or `Q` matrix is required.")
            self.Q = self.G.T @ self.G
        return self

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "json_schema_extra": {"json_encoders": {
            np.ndarray: lambda x: x.ravel().tolist(),
        }}
    }
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Define serialization for numpy arrays."""
        schema = _handler(_source_type)
        
        def serialize_numpy_arrays(obj: Any) -> Any:
            for field_name, field_value in obj.items():
                if isinstance(field_value, np.ndarray):
                    obj[field_name] = field_value.ravel().tolist()
            return obj
            
        schema["serialization"] = core_schema.wrap_serializer_function_ser_schema(
            serialize_numpy_arrays, schema.get("serialization")
        )
        return schema


class ObjectModel(BaseModel):
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

    states: int
    emission: npt.NDArray
    transition: npt.NDArray
    start: npt.NDArray
    name: str = "Default"

    @field_validator("emission", "transition", "start", mode="before")
    @classmethod
    def parse_array(cls, v, info):
        values = info.data
        return np.asarray(v, dtype=float)

    @field_validator("emission", "transition", mode="before")
    @classmethod
    def reshape_emission_transition(cls, v, info):
        values = info.data
        shape = (values["states"], values["states"])
        return np.reshape(v, shape)

    @field_validator("start", mode="before")
    @classmethod
    def reshape_start(cls, v, info):
        values = info.data
        shape = (1, values["states"])
        return np.reshape(v, shape)

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "json_schema_extra": {"json_encoders": {
            np.ndarray: lambda x: x.ravel().tolist(),
        }}
    }
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Define serialization for numpy arrays."""
        schema = _handler(_source_type)
        
        def serialize_numpy_arrays(obj: Any) -> Any:
            for field_name, field_value in obj.items():
                if isinstance(field_value, np.ndarray):
                    obj[field_name] = field_value.ravel().tolist()
            return obj
            
        schema["serialization"] = core_schema.wrap_serializer_function_ser_schema(
            serialize_numpy_arrays, schema.get("serialization")
        )
        return schema


class HypothesisModel(BaseModel):
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

    hypotheses: list[str]
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

    @field_validator("hypotheses", mode="before")
    @classmethod
    def parse_hypotheses(cls, hypotheses):
        if any(h not in H_TYPES for h in hypotheses):
            raise ValueError("Unknown hypothesis type in `hypotheses`.")
        return hypotheses

    def hypotheses_to_generate(self) -> int:
        """Return an integer representation of the hypotheses to generate."""
        h_bin = "".join([str(int(h)) for h in [h in self.hypotheses for h in H_TYPES]])
        return int(h_bin[::-1], 2)

    def as_ctype(self) -> PyHypothesisParams:
        """Return the ctypes representation of the `HypothesisModel`."""
        h_params = PyHypothesisParams()
        fields = [f[0] for f in h_params._fields_]

        for k, v in self.model_dump().items():
            if k in fields:
                setattr(h_params, k, v)

        # set the hypotheses to generate
        h_params.hypotheses_to_generate = self.hypotheses_to_generate()
        return h_params

    model_config = {
        "validate_assignment": True,
        "json_schema_extra": {"json_encoders": {
            np.ndarray: lambda x: x.ravel().tolist(),
        }}
    }
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Define serialization for numpy arrays."""
        schema = _handler(_source_type)
        
        def serialize_numpy_arrays(obj: Any) -> Any:
            for field_name, field_value in obj.items():
                if isinstance(field_value, np.ndarray):
                    obj[field_name] = field_value.ravel().tolist()
            return obj
            
        schema["serialization"] = core_schema.wrap_serializer_function_ser_schema(
            serialize_numpy_arrays, schema.get("serialization")
        )
        return schema
