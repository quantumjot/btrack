#!/usr/bin/env python
# -------------------------------------------------------------------------------
# Name:     btrack
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

from __future__ import annotations

from . import constants

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, NamedTuple

import numpy as np
from numpy import typing as npt

__all__ = ["PyTrackObject", "Tracklet"]


class ImagingVolume(NamedTuple):
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float] | None = None

    @property
    def ndim(self) -> int:
        """Infer the dimensionality from the volume."""
        return (
            constants.Dimensionality.TWO
            if self.z is None
            else constants.Dimensionality.THREE
        )


class PyTrackObject:
    """The base `btrack` track object.

    Primitive class to store information about an object. Essentially a single
    object in a field of view, with some member variables to keep track of data
    associated with an object.

    Parameters
    ----------
    ID : int
        The unique ID of the object.
    x : float
        The x coordinate.
    y : float
        The y coordinate.
    z : float
        The z coordinate.
    t : int
        The timestamp.
    dummy: bool
        Flag for whether the objects is real or a dummy (inserted by the
        tracker when no observation can be linked).
    states : int
        The number of states of the object. This corresponds to the number of
        possible labels.
    label : int
        The label of the object.
    features : array
        A vector of feature values.
    n_features : int
        The length of the feature vector.

    Attributes
    ----------
    properties : dict[str, Union[int, float]]
        Dictionary of properties associated with this object.
    state : constants.States
        A state label for the object. See `constants.States`
    """

    _fields_: ClassVar[list] = [
        ("ID", int),
        ("x", float),
        ("y", float),
        ("z", float),
        ("t", int),
        ("dummy", bool),
        ("states", int),
        ("label", int),
        ("n_features", int),
        ("features", None),
    ]

    def __init__(
        self,
        ID: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        t: int = 0,
        dummy: bool = False,
        states: int = len(constants.States),
        label: int = constants.States.NULL.value,
    ):
        self.ID = int(ID)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.t = int(t)
        self.dummy = bool(dummy)
        self.states = int(states)
        self.label = int(label)
        self.n_features = 0
        self.features = np.zeros(0, dtype=np.float64)
        self._properties: dict[str, Any] = {}

    @property
    def properties(self) -> dict[str, Any]:
        return {} if self.dummy else self._properties

    @properties.setter
    def properties(self, properties: dict[str, Any]):
        """Set the object properties."""
        self._properties.update(properties)

    @property
    def state(self) -> constants.States:
        return constants.States(self.label)

    def set_features(self, keys: list[str]) -> None:
        """Set features to be used by the tracking update."""
        if not keys:
            self.n_features = 0
            self.features = np.zeros(0, dtype=np.float64)
            return

        if any(k not in self.properties for k in keys):
            missing_features = list(set(keys).difference(set(self.properties.keys())))
            raise KeyError(f"Feature(s) missing: {missing_features}.")

        self._features = np.concatenate(
            [np.asarray(self.properties[k]).ravel() for k in keys], axis=-1
        ).astype(np.float64)

        self.features = self._features
        self.n_features = len(self._features)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary of the fields and their values."""
        node = {
            "ID": self.ID,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "t": self.t,
            "dummy": self.dummy,
            "states": self.states,
            "label": self.label,
        }
        node |= self.properties
        return node

    @classmethod
    def from_dict(cls, properties: dict[str, Any]) -> PyTrackObject:
        """Build an object from a dictionary."""
        obj = cls()
        fields = {"ID", "x", "y", "z", "t", "dummy", "states", "label"}
        for key in fields.intersection(properties):
            new_data = properties[key]
            if key in ("ID", "t", "states", "label"):
                setattr(obj, key, int(new_data))
            elif key in ("dummy",):
                setattr(obj, key, bool(new_data))
            else:
                setattr(obj, key, float(new_data))

        # we can add any extra details to the properties dictionary
        obj.properties = {k: v for k, v in properties.items() if k not in fields}
        return obj

    def __repr__(self):
        return self.to_dict().__repr__()

    def _repr_html_(self):
        return _pandas_html_repr(self)


@dataclass
class PyTrackingInfo:
    """Primitive class to store information about the tracking output.

    Parameters
    ----------
    error : int
        Error code from the tracker. See `constants.Errors` for definitions.
    n_tracks : int
        Total number of tracks initialised during tracking.
    n_active : int
        Number of active tracks.
    n_conflicts : int
        Number of conflicts.
    n_lost : int
        Number of lost tracks.
    t_update_belief : float
        Time to update belief matrix in ms.
    t_update_link : float
        Time to update links in ms.
    t_total_time : float
        Total time to track objects.
    p_link : float
        Typical probability of association.
    p_lost : float
        Typical probability of losing track.
    complete : bool
        Flag denoting that the tracking is complete.
    """

    error: int = 910
    n_tracks: int = 0
    n_active: int = 0
    n_conflicts: int = 0
    n_lost: int = 0
    t_update_belief: float = 0.0
    t_update_link: float = 0.0
    t_total_time: float = 0.0
    p_link: float = 0.0
    p_lost: float = 0.0
    complete: bool = False

    _fields_: ClassVar[list] = [
        ("error", int),
        ("n_tracks", int),
        ("n_active", int),
        ("n_conflicts", int),
        ("n_lost", int),
        ("t_update_belief", float),
        ("t_update_link", float),
        ("t_total_time", float),
        ("p_link", float),
        ("p_lost", float),
        ("complete", bool),
    ]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary of the statistics"""
        return asdict(self)

    @property
    def tracker_active(self) -> bool:
        """Return the current status."""
        no_error = constants.Errors(self.error) == constants.Errors.NO_ERROR
        return no_error and not self.complete


@dataclass
class PyGraphEdge:
    """A structure defining an edge in the association graph. This is derived
    from the Bayesian belief matrix in the initial step of the tracking
    algorithm.

    Parameters
    ----------
    source : int
        A reference to a source object.
    target : int
        A reference to a target object.
    score : float
        The posterior probability of linking the target object to the source.
    edge_type : int
        Type of graph edge.
    """

    source: int = 0
    target: int = 0
    score: float = 0.0
    edge_type: int = 0

    _fields_: ClassVar[list] = [
        ("source", int),
        ("target", int),
        ("score", float),
        ("edge_type", int),
    ]

    @property
    def probability(self) -> float:
        return self.score

    @probability.setter
    def probability(self, value: float) -> None:
        self.score = float(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary describing the edge."""
        return asdict(self)


class Tracklet:
    """A `btrack` Tracklet object used to store track information.

    Parameters
    ----------
    ID : int
        A unique integer identifier for the tracklet.
    data : list[PyTrackObject]
        The objects linked together to form the track.
    parent : int
        The identifiers of the parent track(s).
    children : list
        The identifiers of the child tracks.
    fate : constants.Fates, default = constants.Fates.UNDEFINED
        An enumerated type describing the fate of the track.

    Attributes
    ----------
    x : list[float]
        The list of x positions.
    y : list[float]
        The list of y positions.
    z : list[float]
        The list of z positions.
    t : list[float]
        The list of timestamps.
    dummy : list[bool]
        A list specifying which objects are dummy objects inserted by the tracker.
    parent : int, list
        The identifiers of the parent track(s).
    generation : int
        If specified, the generational depth of the tracklet releative to the root.
    refs : list[int]
        Returns a list of :py:class:`btrack.btypes.PyTrackObject` identifiers
        used to build the track. Useful for indexing back into the original
        data, e.g. table of localizations or h5 file.
    label : list[str]
        Return the label of each object in the track.
    state : list[int]
        Return the numerical label of each object in the track.
    softmax : list[float]
        If defined, return the softmax score for the label of each object in the
        track.
    properties : dict[str, npt.NDArray]
        Return a dictionary of track properties derived from
        :py:class:`btrack.btypes.PyTrackObject` properties.
    root : int,
        The identifier of the root ID if a branching tree (ie cell division).
    is_root : boole
        Flag to denote root track.
    is_leaf : bool
        Flag to denote leaf track.
    start : int, float
        First time stamp of track.
    stop : int, float
        Last time stamp of track.
    kalman : npt.NDArray
        Return the complete output of the kalman filter for this track. Note,
        that this may not have been returned while from the tracker. See
        :py:attr:`btrack.BayesianTracker.return_kalman` for more details.
    LBEP : list
        An LBEP representation of the track.


    Notes
    -----
    Tracklet object for storing and updating linked lists of track objects.
    Forms the data structure for an individual tracklet. Track 'fates' are the
    selected hypotheses after optimization. Defined in constants.Fates. Intrinsic
    properties can be accesses as attributes, e.g: track.x returns the track
    x values.
    """

    def __init__(
        self,
        ID: int,
        data: list[PyTrackObject],
        *,
        parent: int | None = None,
        children: list[int] | None = None,
        fate: constants.Fates = constants.Fates.UNDEFINED,
    ):
        assert all(isinstance(o, PyTrackObject) for o in data)

        self.ID = ID
        self._data = data
        self._kalman = np.empty(0)

        self.root = None
        self.parent = parent
        self.children = children if children is not None else []
        self.type = None
        self.fate = fate
        self.generation = 0

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self.to_dict().__repr__()

    def _repr_html_(self):
        return _pandas_html_repr(self)

    @property
    def properties(self) -> dict:
        """Return the properties of the objects."""
        # find the set of keys, then grab the properties
        keys: set = set()
        for obj in self._data:
            keys.update(obj.properties.keys())

        # work out the shapes of the properties by finding the first object that
        # is not a dummy and returning the shape of the property, we can use
        # this to fill the properties array with NaN for dummy objects
        property_shapes = {
            k: next(
                (np.asarray(o.properties[k]).shape for o in self._data if not o.dummy),
                None,
            )
            for k in keys
        }

        # set the properties, replacing missing values with a NaN
        properties = {
            k: [
                (
                    o.properties[k]
                    if k in o.properties
                    else np.full(property_shapes[k], np.nan)
                )
                for o in self._data
            ]
            for k in keys
        }

        # validate the track properties
        for k, v in properties.items():
            if len(v) != len(self):
                raise ValueError(
                    "The number of properties and track objects must be equal."
                )
            # ensure the property values are a numpy array
            if not isinstance(v, np.ndarray):
                properties[k] = np.asarray(v)

        return properties

    @properties.setter
    def properties(self, properties: dict[str, npt.NDArray]):
        """Store properties associated with this Tracklet."""
        # TODO(arl): this will need to set the object properties
        pass

    def __getitem__(self, attr: str):
        assert isinstance(attr, str)
        try:
            return getattr(self, attr)
        except AttributeError:
            return self.properties[attr]

    @property
    def x(self) -> list:
        return [o.x for o in self._data]

    @property
    def y(self) -> list:
        return [o.y for o in self._data]

    @property
    def z(self) -> list:
        return [o.z for o in self._data]

    @property
    def t(self) -> list:
        return [o.t for o in self._data]

    @property
    def dummy(self) -> list:
        return [o.dummy for o in self._data]

    @property
    def refs(self) -> list:
        return [o.ID for o in self._data]

    @property
    def start(self) -> list:
        return self.t[0]

    @property
    def stop(self) -> list:
        return self.t[-1]

    @property
    def label(self) -> list:
        return [o.state.name for o in self._data]

    @property
    def state(self) -> list:
        return [o.state.value for o in self._data]

    @property
    def softmax(self) -> list:
        return [o.probability for o in self._data]

    @property
    def is_root(self) -> bool:
        return self.parent == 0 or self.parent is None or self.parent == self.ID

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def kalman(self) -> npt.NDArray:
        return self._kalman

    @kalman.setter
    def kalman(self, data: npt.NDArray) -> None:
        assert isinstance(data, np.ndarray)
        self._kalman = data

    def mu(self, index: int) -> npt.NDArray:
        """Return the Kalman filter mu. Note that we are only returning the mu
        for the positions (e.g. 3x1)."""
        return self.kalman[index, 1:4].reshape(3, 1)

    def covar(self, index: int) -> npt.NDArray:
        """Return the Kalman filter covariance matrix. Note that we are
        only returning the covariance matrix for the positions (e.g. 3x3)."""
        return self.kalman[index, 4:13].reshape(3, 3)

    def predicted(self, index: int) -> npt.NDArray:
        """Return the motion model prediction for the given timestep."""
        return self.kalman[index, 13:].reshape(3, 1)

    def to_dict(
        self, properties: list = constants.DEFAULT_EXPORT_PROPERTIES
    ) -> dict[str, Any]:
        """Return a dictionary of the tracklet which can be used for JSON
        export. This is an ordered dictionary for nicer JSON output.
        """
        trk_tuple = tuple((p, getattr(self, p)) for p in properties)
        data = OrderedDict(trk_tuple)
        data |= self.properties
        return data

    def to_array(
        self, properties: list = constants.DEFAULT_EXPORT_PROPERTIES
    ) -> npt.NDArray:
        """Return a representation of the trackled as a numpy array."""
        data = self.to_dict(properties)
        tmp_track = []
        for values in data.values():
            np_values = np.asarray(values)
            if np_values.size == 1:
                np_values = np.tile(np_values, len(self))
            np_values = np.reshape(np_values, (len(self), -1))
            tmp_track.append(np_values)

        tmp_track_arr = np.concatenate(tmp_track, axis=-1)
        assert tmp_track_arr.shape[0] == len(self)
        assert tmp_track_arr.ndim == constants.Dimensionality.TWO
        return tmp_track_arr.astype(np.float32)

    def in_frame(self, frame: int) -> bool:
        """Return true or false as to whether the track is in the frame."""
        return self.t[0] <= frame and self.t[-1] >= frame

    def trim(self, frame: int, tail: int = 75) -> Tracklet:
        """Trim the tracklet and return one with the trimmed data."""
        d = [o for o in self._data if o.t <= frame and o.t >= frame - tail]
        return Tracklet(self.ID, d)

    def LBEP(self) -> tuple[int, list, list, int | None, None, int]:
        """Return an LBEP table summarising the track."""
        return (
            self.ID,
            self.start,
            self.stop,
            self.parent,
            self.root,
            self.generation,
        )


def _pandas_html_repr(obj):
    """Prepare data for HTML representation in a notebook."""
    try:
        import pandas as pd
    except ImportError:
        return (
            "<b>Install pandas for nicer, tabular rendering.</b> <br>" + obj.__repr__()
        )

    obj_as_dict = obj.to_dict()

    # now try to process for display in the notebook
    n_items = len(obj) if hasattr(obj, "__len__") else 1

    for k, v in obj_as_dict.items():
        if not isinstance(v, (list, np.ndarray)):
            obj_as_dict[k] = [v] * n_items
        elif isinstance(v, np.ndarray):
            ndim = 0 if n_items == 1 else 1
            if v.ndim > ndim:
                obj_as_dict[k] = [f"{v.shape[ndim:]} array"] * n_items

    return pd.DataFrame.from_dict(obj_as_dict).to_html()
