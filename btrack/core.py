from __future__ import annotations

import ctypes
import itertools
import logging
import os
import warnings

import numpy as np
from numpy import typing as npt

from btrack import btypes, config, constants, libwrapper, models, utils
from btrack.io.exporters import export_delegator
from btrack.io.utils import localizations_to_objects
from btrack.optimise import hypothesis, optimiser

__version__ = constants.get_version()

# get the logger instance
logger = logging.getLogger(__name__)


class BayesianTracker:
    """BayesianTracker.

    BayesianTracker is a multi object tracking algorithm, specifically used to
    reconstruct tracks in crowded fields. Here we use a probabilistic network of
    information to perform the trajectory linking. This class is a wrapper for
    the C++ implementation of the BayesianTracker.

    Parameters
    ----------
    verbose : bool
        A flag to set the verbosity level while logging the output.

    Attributes
    ----------
    n_tracks : int
        The number of found tracks.
    n_dummies : int
        The number of inserted dummy objects.
    tracks : list
        A list of Tracklet objects.
    refs : list
        References to the objects forming the tracks.
    dummies : list
        The dummy objects inserted by the tracker.
    volume : tuple
        The imaging volume as [(xlo, xhi), ..., (zlo, zhi)]. See
        :py:meth:`btrack.btypes.ImagingVolume` for more details.
    frame_range : tuple
        The frame range for tracking, essentially the last dimension of volume.
    LBEP : List[List]
        Return an LBEP table of the track lineages.
    configuration : config.TrackerConfig
        Return the current configuration.

    Notes
    -----
    This method uses positional information (position, velocity ...) as well as
    visual information (labels, features...) for track linking.

    The tracking algorithm assembles reliable sections of track that do not
    contain splitting events (tracklets). Each new tracklet initiates a
    probabilistic model in the form of a Kalman filter [1]_, and utilises this to
    predict future states (and error in states) of each of the objects in the
    field of view.  We assign new observations to the growing tracklets
    (linking) by evaluating the posterior probability of each potential linkage
    from a Bayesian belief matrix for all possible linkages [2]_. The best
    linkages are those with the highest posterior probability.

    Data can be passed in in the following formats:

        * numpy arrays
        * :py:class:`btrack.btypes.PyTrackObject`
        * CSV (see :py:meth:`btrack.io.importers.import_CSV`)
        * HDF (see :py:class:`btrack.io.hdf.HDF5FileHandler`)

    The tracker can be used to return all of the original data neatly packaged
    into tracklet objects, or as a nested list of references to the original
    data sets. The latter is useful if using only the first part of a tracking
    protocol, or other metadata is needed for further analysis. The references
    can be used to make symbolic links in HDF5 files, for example. Use
    `optimise` to generate hypotheses for global optimisation [3]_ [4]_. Read the
    :py:class:`optimiser.TrackOptimiser` documentation for more information about the
    track linker.

    Full details of the implementation can be found in [5]_ and [6]_.

    Examples
    --------
    Can be used with ContextManager support, like this:

        >>> with BayesianTracker() as tracker:
        >>>    tracker.configure("./models/cell_config.json")
        >>>    tracker.append(observations)
        >>>    tracker.track()
        >>>    tracks = tracker.tracks

    References
    ----------
    .. [1] 'A new approach to linear filtering and prediction problems.'
      Kalman RE, 1960 Journal of Basic Engineering

    .. [2] 'A Bayesian algorithm for tracking multiple moving objects in outdoor
      surveillance video', Narayana M and Haverkamp D 2007 IEEE

    .. [3] 'Report Automated Cell Lineage Construction' Al-Kofahi et al.
      Cell Cycle 2006 vol. 5 (3) pp. 327-335

    .. [4] 'Reliable cell tracking by global data association', Bise et al.
      2011 IEEE Symposium on Biomedical Imaging pp. 1004-1010

    .. [5] 'Local cellular neighbourhood controls proliferation in cell
      competition', Bove A, Gradeci D, Fujita Y, Banerjee S, Charras G and
      Lowe AR 2017 Mol. Biol. Cell vol 28 pp. 3215-3228

    .. [6] 'Automated deep lineage tree analysis using a Bayesian single cell
      tracking approach', Ulicna K, Vallardi G, Charras G and Lowe AR 2021 Front.
      Comput. Sci. 3
    """

    def __init__(
        self,
        *,
        verbose: bool = True,
    ):
        """Initialise the BayesianTracker C++ engine and parameters."""

        # load the library, get an instance of the engine
        self._initialised = False
        self._lib = libwrapper.get_library()
        self._engine = self._lib.new_interface(verbose)

        if not verbose:
            logger.setLevel(logging.WARNING)

        # store a default config
        self._config = config.TrackerConfig(verbose=verbose)

        # sanity check library version
        version_tuple = constants.get_version_tuple()
        if not self._lib.check_library_version(self._engine, *version_tuple):
            logger.warning(f"btrack (v{__version__}) shared library mismatch.")
        else:
            logger.info(f"btrack (v{__version__}) library imported")

        # silently set the update method to EXACT
        self._lib.set_update_mode(self._engine, self.configuration.update_method.value)

        # default parameters and space for stored objects
        self._objects: list[btypes.PyTrackObject] = []
        self._frame_range = [0, 0]

    def __enter__(self):
        logger.info("Starting BayesianTracker session")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info("Ending BayesianTracker session")
        self._lib.del_interface(self._engine)

    def configure_from_file(self, filename: os.PathLike) -> None:
        """Configure the tracker from a configuration file. See `configure`."""
        warnings.warn(
            "This function will be deprecated. Use `.configure()` instead.",
            DeprecationWarning,
        )
        self.configure(filename)

    def configure(
        self, configuration: dict | os.PathLike | config.TrackerConfig
    ) -> None:
        """Configure the tracker with a motion model, an object model and
        hypothesis generation_parameters.

        Parameters
        ----------
        configuration :
            A dictionary containing the configuration options for a tracking
            session.
        """
        if isinstance(configuration, dict):
            configuration = config.TrackerConfig(**configuration)
        elif isinstance(configuration, (str, os.PathLike)):
            configuration = config.load_config(configuration)

        self._config = configuration

        # set all configuration options using setattr
        for attr in configuration.__fields__:
            setattr(self, attr, getattr(configuration, attr))

        self._initialised = True

    @property
    def configuration(self) -> config.TrackerConfig:
        """Get the current configuration."""
        return self._config

    def __getattr__(self, attr):
        """Default to config if we do not have a specific getter/setter."""
        return getattr(self.configuration, attr)

    def __setattr__(self, attr, value):
        if not attr.startswith("_") and self.configuration.verbose:
            logger.info(f"Setting {attr} -> {value}")

        if attr in config.TrackerConfig.__fields__:
            setattr(self.configuration, attr, value)
        else:
            object.__setattr__(self, attr, value)

        # if we need to update the C++ library instance, do it here
        if (
            attr
            in (
                "motion_model",
                "object_model",
                "max_search_radius",
                "volume",
                "update_method",
            )
            and value is not None
        ):
            update_lib_func = getattr(self, f"_{attr}")
            update_lib_func(value)

    def __len__(self) -> int:
        return self.n_tracks

    def _max_search_radius(self, max_search_radius: int):
        """Set the maximum search radius for fast cost updates."""
        self._lib.max_search_radius(self._engine, max_search_radius)

    def _update_method(self, method: str | constants.BayesianUpdates):
        """Set the method for updates, EXACT, APPROXIMATE, CUDA etc..."""
        self._lib.set_update_mode(self._engine, method.value)

    @property
    def n_tracks(self) -> int:
        """Return the number of tracks found."""
        return self._lib.size(self._engine)

    @property
    def n_dummies(self) -> int:
        """Return the number of dummy objects (negative ID)."""
        return len([d for d in itertools.chain.from_iterable(self.refs) if d < 0])

    @property
    def tracks(self) -> list[btypes.Tracklet]:
        """Return a sorted list of tracks, default is to sort by increasing
        length."""
        return [self[i] for i in range(self.n_tracks)]

    @property
    def refs(self):
        """Return tracks as a list of IDs (essentially pointers) to the original
        objects. Use this to write out HDF5 tracks."""
        tracks = []
        for i in range(self.n_tracks):
            # get the track length
            n = self._lib.track_length(self._engine, i)

            # set up some space for the output and  get the track data
            refs = np.zeros((n,), dtype=np.int32)
            _ = self._lib.get_refs(self._engine, refs, i)
            tracks.append(refs.tolist())

        return tracks

    @property
    def dummies(self):
        """Return a list of dummy objects."""
        return [
            self._lib.get_dummy(self._engine, -(i + 1)) for i in range(self.n_dummies)
        ]

    @property
    def LBEP(self):
        """Return an LBEP list describing the track lineage information.

        Notes
        -----

        .. list-table:: LBEP table layout
           :widths: 10 90
           :header-rows: 1

           * - Index
             - Description
           * - L
             - A unique label of the track (label of markers, 16-bit positive).
           * - B
             - A zero-based temporal index of the frame in which the track begins.
           * - E
             - A zero-based temporal index of the frame in which the track ends.
           * - P
             - Label of the parent track (0 is used when no parent is defined).
           * - R
             - Label of the root track.
           * - G
             - Generational depth (from root).
        """
        return utils._lbep_table(self.tracks)

    def _sort(self, tracks: list[btypes.Tracklet]) -> list[btypes.Tracklet]:
        """Return a sorted list of tracks"""
        return sorted(tracks, key=lambda t: len(t), reverse=True)

    def _volume(self, volume: tuple | btypes.ImagingVolume) -> None:
        """Set the imaging volume.

        Parameters
        ----------
        volume : tuple, ImagingVolume
            A tuple describing the imaging volume.
        """
        volume = btypes.ImagingVolume(*volume)

        # if we've only provided 2 dims, set the last one to a default
        if volume.ndim == constants.Dimensionality.TWO:
            z = (-1e5, 1e5)
            volume = btypes.ImagingVolume(volume.x, volume.y, z)

        self._lib.set_volume(self._engine, np.array(volume, dtype=float))

    def _motion_model(self, model: models.MotionModel) -> None:
        """Set a new motion model. Must be of type MotionModel, either loaded
        from file or instantiating a MotionModel.

        Parameters
        ----------
        model : MotionModel
            A motion model to be used by the tracker.
        """

        # need to populate fields in the C++ library
        self._lib.motion(
            self._engine,
            model.measurements,
            model.states,
            model.A,
            model.H,
            model.P,
            model.Q,
            model.R,
            model.dt,
            model.accuracy,
            model.max_lost,
            model.prob_not_assign,
        )

    def _object_model(self, model: models.ObjectModel) -> None:
        """Set a new object model. Must be of type ObjectModel, either loaded
        from file or instantiating an ObjectModel.

        Parameters
        ----------
        model : ObjectModel
        """
        # if model is not None:
        logger.info(f"Loading object model: {model.name}")
        self.configuration.object_model = model

        # need to populate fields in the C++ library
        self._lib.model(
            self._engine,
            model.states,
            model.emission,
            model.transition,
            model.start,
        )

    @property
    def frame_range(self) -> tuple[int, int]:
        """Return the frame range."""
        return tuple(self.configuration.frame_range)

    @property
    def objects(self) -> list[btypes.PyTrackObject]:
        """Return the list of objects added through the append method."""
        return self._objects

    def append(self, objects: list[btypes.PyTrackObject] | npt.NDArray) -> None:
        """Append a single track object, or list of objects to the stack. Note
        that the tracker will automatically order these by frame number, so the
        order here does not matter. This means several datasets can be
        concatenated easily, by running this a few times.

        Parameters
        ----------
        objects : list, npt.NDArray
            A list of objects to track.
        """

        objects = localizations_to_objects(objects)

        for idx, obj in enumerate(objects):
            if not isinstance(obj, btypes.PyTrackObject):
                raise TypeError("track_object must be a `PyTrackObject`")

            obj.ID = idx + len(self._objects)  # make sure ID tracks properly

            # call the `obj.set_features` method for each object, which will
            # set the pointer to the array for the tracker
            obj.set_features(self.configuration.features)

            self._frame_range[1] = max(obj.t, self._frame_range[1])
            _ = self._lib.append(self._engine, obj)

        # store a copy of the list of objects
        self._objects += objects

    def _stats(self, info_ptr: ctypes.pointer) -> btypes.PyTrackingInfo:
        """Cast the info pointer back to an object."""

        if not isinstance(info_ptr, ctypes.POINTER(btypes.PyTrackingInfo)):
            raise TypeError("Stats requires the pointer to the object")

        return info_ptr.contents

    def track_interactive(self, *args, **kwargs) -> None:
        logger.warning("`track_interactive` will be deprecated. Use `track` instead.")
        return self.track(*args, **kwargs)

    def track(
        self,
        *,
        step_size: int = 100,
        tracking_updates: list[str | constants.BayesianUpdateFeatures] | None = None,
    ) -> None:
        """Run the tracking in an interactive mode.

        Parameters
        ----------
        step_size : int, default=100
            The number of tracking steps to be taken before returning summary
            statistics. The tracking will be followed to completion, regardless
            of the step size provided.
        tracking_updates : list, optional
            A list of tracking updates to perform. See
            :py:class:`btrack.btypes.BayesianUpdateFeatures` for details.
        """

        logger.info("Starting tracking... ")

        if tracking_updates is not None:
            self.configuration.tracking_updates = tracking_updates

        logger.info(
            f"Update using: {[f.name for f in self.configuration.tracking_updates]}"
        )

        # bitwise OR is equivalent to int sum here
        self._lib.set_update_features(
            self._engine,
            sum(int(f.value) for f in self.configuration.tracking_updates),
        )

        stats = self.step()
        frame = 0

        # while not stats.complete and stats.error not in constants.ERRORS:
        while stats.tracker_active:
            logger.info(
                f"Tracking objects in frames {frame} to "
                f"{min(frame+step_size-1, self._frame_range[1]+1)} "
                f"(of {self._frame_range[1]+1})..."
            )

            stats = self.step(step_size)
            utils.log_stats(stats.to_dict())
            frame += step_size

        if not utils.log_error(stats.error):
            logger.info("SUCCESS.")
            logger.info(
                f" - Found {self.n_tracks} tracks in "
                f"{1+self._frame_range[1]} frames "
                f"(in {stats.t_total_time}s)"
            )
            logger.info(
                f" - Inserted {self.n_dummies} dummy objects to fill tracking gaps"
            )

    def step(self, n_steps: int = 1) -> btypes.PyTrackingInfo | None:
        """Run an iteration (or more) of the tracking. Mostly for interactive
        mode tracking."""
        return (
            self._stats(self._lib.step(self._engine, n_steps))
            if self._initialised
            else None
        )

    def hypotheses(self) -> list[hypothesis.Hypothesis]:
        """Calculate and return hypotheses using the hypothesis engine."""

        if not self.hypothesis_model:
            raise AttributeError("Hypothesis model has not been specified.")

        n_hypotheses = self._lib.create_hypotheses(
            self._engine,
            self.hypothesis_model.as_ctype(),
            self._frame_range[0],
            self._frame_range[1],
        )

        return [self._lib.get_hypothesis(self._engine, h) for h in range(n_hypotheses)]

    def optimize(self, **kwargs):
        """Proxy for `optimise` for our American friends ;)"""
        return self.optimise(**kwargs)

    def optimise(self, options: dict | None = None) -> list[hypothesis.Hypothesis]:
        """Optimize the tracks.

        Parameters
        ----------
        options : dict
            A set of options to be used by GLPK during convex optimization.

        Returns
        -------
        optimized : list
            The list of hypotheses which represents the optimal solution.

        Notes
        -----
        This generates the hypotheses for track merges, branching etc, runs the
        optimiser and then performs track merging, removal of track fragments,
        renumbering and assignment of branches.
        """
        logger.info(f"Loading hypothesis model: {self.hypothesis_model.name}")

        logger.info(f"Calculating hypotheses (relax: {self.hypothesis_model.relax})...")
        hypotheses = self.hypotheses()

        # if we have not been provided with optimizer options, use the default
        # from the configuration.
        options = options or self.configuration.optimizer_options

        # if we don't have any hypotheses return
        if not hypotheses:
            logger.warning("No hypotheses could be found.")
            return []

        # set up the track optimiser
        track_linker = optimiser.TrackOptimiser(options=options)
        track_linker.hypotheses = hypotheses
        selected_hypotheses = track_linker.optimise()
        optimised = [hypotheses[i] for i in selected_hypotheses]

        if not optimised:
            logger.warning("Optimization failed.")
            return []

        h_original = [h.hypothesis_type for h in hypotheses]
        h_optimise = [h.hypothesis_type for h in optimised]
        h_types = sorted(set(h_original), key=lambda h: h.value)

        for h_type in h_types:
            logger.info(
                f" - {h_type}: {h_optimise.count(h_type)}"
                f" (of {h_original.count(h_type)})"
            )
        logger.info(f" - TOTAL: {len(hypotheses)} hypotheses")

        # now that we have generated the optimal sequence, merge all of the
        # tracks, delete fragments and assign divisions
        h_array = np.array(selected_hypotheses, dtype=np.uint32)
        h_array = h_array[np.newaxis, ...]
        self._lib.merge(self._engine, h_array, len(selected_hypotheses))
        logger.info(f"Completed optimization with {self.n_tracks} tracks")

        return optimised

    def __getitem__(self, idx: int) -> btypes.Tracklet:
        """Return a single track from the BayesianTracker object."""
        # get the track length
        n = self._lib.track_length(self._engine, idx)

        # set up some space for the output
        children = np.zeros((2,), dtype=np.int32)  # pointers to children
        refs = np.zeros((n,), dtype=np.int32)  # pointers to objects

        # get the track data
        _ = self._lib.get_refs(self._engine, refs, idx)
        nc = self._lib.get_children(self._engine, children, idx)
        p = self._lib.get_parent(self._engine, idx)
        f = constants.Fates(self._lib.get_fate(self._engine, idx))

        # get the track ID
        trk_id = self._lib.get_ID(self._engine, idx)

        # convert the array of children to a python list
        c = children.tolist() if nc > 0 else []

        # now build the track from the references
        refs = refs.tolist()
        dummies = [self._lib.get_dummy(self._engine, d) for d in refs if d < 0]

        track = []
        for r in refs:
            if r < 0:
                # TODO(arl): softmax scores are zero for dummy objects
                dummy = dummies.pop(0)
                dummy.probability = np.zeros((5,), dtype=np.float32)
                track.append(dummy)
            else:
                track.append(self._objects[r])

        # make a new track object and return it
        trk = btypes.Tracklet(trk_id, track, parent=p, children=c, fate=f)
        trk.root = self._lib.get_root(self._engine, idx)
        trk.generation = self._lib.get_generation(self._engine, idx)

        if not self.return_kalman:
            return trk

        # get the size of the Kalman arrays
        sz_mu = self.motion_model.measurements + 1
        sz_cov = self.motion_model.measurements**2 + 1

        # otherwise grab the kalman filter data
        kal_mu = np.zeros((n, sz_mu), dtype=np.float64)  # kalman filtered
        kal_cov = np.zeros((n, sz_cov), dtype=np.float64)  # kalman covariance
        kal_pred = np.zeros((n, sz_mu), dtype=np.float64)  # motion predict

        _ = self._lib.get_kalman_mu(self._engine, kal_mu, idx)
        _ = self._lib.get_kalman_covar(self._engine, kal_cov, idx)
        _ = self._lib.get_kalman_pred(self._engine, kal_pred, idx)

        # cat the data [mu(0),...,mu(n),cov(0,0),...cov(n,n), pred(0),..]
        trk.kalman = np.hstack((kal_mu, kal_cov[:, 1:], kal_pred[:, 1:]))
        return trk

    def export(
        self,
        filename: os.PathLike,
        obj_type: str | None = None,
        filter_by: str | None = None,
    ) -> None:
        """Export tracks using the appropriate exporter.

        Parameters
        ----------
        filename : str
            The filename to export the data. The extension (e.g. .h5) is used
            to select the correct export function.
        obj_type : str, optional
            The object type to export the data. Usually `obj_type_1`
        filter_by : str, optional
            A string that represents how the data has been filtered prior to
            tracking, e.g. using the object property `area>100`
        """
        export_delegator(filename, self, obj_type=obj_type, filter_by=filter_by)

    def to_napari(
        self,
        *,
        replace_nan: bool = True,
        ndim: int | None = None,
    ) -> tuple[npt.NDArray, dict, dict]:
        """Return the data in a format for a napari tracks layer.
        See :py:meth:`btrack.utils.tracks_to_napari`."""

        ndim = self.configuration.volume.ndim if ndim is None else ndim

        return utils.tracks_to_napari(self.tracks, ndim=ndim, replace_nan=replace_nan)
