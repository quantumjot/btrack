#!/usr/bin/env python
# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------

__author__ = "Alan R. Lowe"
__email__ = "a.lowe@ucl.ac.uk"

import ctypes
import itertools
import logging

import numpy as np

from . import btypes, constants, libwrapper, utils
from .dataio import export_delegator, localizations_to_objects
from .optimise import optimiser

__version__ = constants.get_version()

# get the logger instance
logger = logging.getLogger('worker_process')

# if we don't have any handlers, set one up
if not logger.handlers:
    # configure stream handler
    log_fmt = logging.Formatter(
        '[%(levelname)s][%(asctime)s] %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p',
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_fmt)

    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)


class BayesianTracker:
    """BayesianTracker.

    BayesianTracker is a multi object tracking algorithm, specifically
    used to reconstruct tracks in crowded fields. Here we use a probabilistic
    network of information to perform the trajectory linking.

    Parameters
    ----------
    verbose : bool
        A flag to set the verbosity level while logging the output.
    max_search_radius : int, float
        The maximum search radius of the algorithm in isotropic units of the
        data. Should be greater than zero.

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
        The imaging volume as [(xlo, xhi), ..., (zlo, zhi), (tlo, thi)]
    frame_range : tuple
        The frame range for tracking, essentially the last dimension of volume.
    max_search_radius : int, float
        The maximum search radius of the algorithm in isotropic units of the
        data. Should be greater than zero.
    motion_model : MotionModel
        A motion model to make motion predictions.
    object_model : ObjectModel
        An object model to make state predictions.
    update_method : BayesianUpdates
        The method to perform the bayesian updates during tracklet linking.
            BayesianUpdates.EXACT
                Use the exact Bayesian update method. Can be slow for systems
                with many objects.
            BayesianUpdates.APPROXIMATE
                Use the approximate Bayesian update method. Useful for systems
                with may objects.
            BayesianUpdates.CUDA
                Use the CUDA implementation of the Bayesian update method. Not
                currently implemented.
    return_kalman : bool
        Flag to request the Kalman debug info when returning tracks.
    lbep :
        Return an LBEP table of the track lineages.

    Notes
    -----
    This method uses positional information (position, velocity ...) as well as
    visual information (labels, features...) for track linking.

    The tracking algorithm assembles reliable sections of track that do not
    contain splitting events (tracklets). Each new tracklet initiates a
    probabilistic model in the form of a Kalman filter (Kalman, 1960), and
    utilises this to predict future states (and error in states) of each of the
    objects in the field of view.  We assign new observations to the growing
    tracklets (linking) by evaluating the posterior probability of each
    potential linkage from a Bayesian belief matrix for all possible linkages
    (Narayana and Haverkamp, 2007). The best linkages are those with the
    highest posterior probability.

    This class is a wrapper for the C++ implementation of the BayesianTracker.

    Data can be passed in in the following formats:
        - btrack PyTrackObject (defined in btypes)
        - Optional JSON files using loaders
        - HDF

    Can be used with ContextManager support, like this:

        >>> with BayesianTracker() as tracker:
        >>>    tracker.append(observations)
        >>>    tracker.track()

    The tracker can be used to return all of the original data neatly packaged
    into tracklet objects, or as a nested list of references to the original
    data sets. The latter is useful if using only the first part of a tracking
    protocol, or other metadata is needed for further analysis. The references
    can be used to make symbolic links in HDF5 files, for example.

    Use the .tracks to return Tracklets, or .refs to return the references.

    Use optimise to generate hypotheses for global optimisation. Read the
    TrackLinker documentation for more information about the track linker.

    References
    ----------
    'A Bayesian algorithm for tracking multiple moving objects in outdoor
    surveillance video', Narayana M and Haverkamp D 2007 IEEE

    'Report Automated Cell Lineage Construction' Al-Kofahi et al.
    Cell Cycle 2006 vol. 5 (3) pp. 327-335

    'Reliable cell tracking by global data association', Bise et al.
    2011 IEEE Symposium on Biomedical Imaging pp. 1004-1010

    'Local cellular neighbourhood controls proliferation in cell
    competition', Bove A, Gradeci D, Fujita Y, Banerjee S, Charras G and
    Lowe AR 2017 Mol. Biol. Cell vol 28 pp. 3215-3228
    """

    def __init__(
        self,
        verbose: bool = True,
        max_search_radius: int = constants.MAX_SEARCH_RADIUS,
    ):
        """Initialise the BayesianTracker C++ engine and parameters."""

        # load the library, get an instance of the engine
        self._initialised = False
        self._lib = libwrapper.get_library()
        self._engine = self._lib.new_interface(verbose)

        if not verbose:
            logger.setLevel(logging.WARNING)

        # sanity check library version
        version_tuple = constants.get_version_tuple()
        if not self._lib.check_library_version(self._engine, *version_tuple):
            logger.warning(f'btrack (v{__version__}) shared library mismatch.')
        else:
            logger.info(f'btrack (v{__version__}) library imported')

        # silently set the update method to EXACT
        self._bayesian_update_method = constants.BayesianUpdates.EXACT
        self._lib.set_update_mode(self._engine, self.update_method.value)

        # default parameters and space for stored objects
        self._objects = []
        self._motion_model = None
        self._object_model = None
        self._frame_range = [0, 0]
        self.max_search_radius = max_search_radius
        self.return_kalman = False

    def __enter__(self):
        logger.info('Starting BayesianTracker session')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info('Ending BayesianTracker session')
        self._lib.del_interface(self._engine)

    def configure_from_file(self, filename: str):
        """Configure the tracker from a configuration file. See `configure`."""
        config = utils.load_config(filename)
        self.configure(config)

    def configure(self, config: dict):
        """Configure the tracker with a motion model, an object model and
        hypothesis generation_parameters.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration options for a tracking
            session.
        """

        if not isinstance(config, dict):
            raise TypeError('configuration must be a dictionary')

        # store the models locally
        self.motion_model = config.get("MotionModel", None)
        self.object_model = config.get("ObjectModel", None)
        self.hypothesis_model = config.get("HypothesisModel", None)
        self._initialised = True

    def __len__(self):
        return self.n_tracks

    @property
    def max_search_radius(self):
        return self._max_search_radius

    @max_search_radius.setter
    def max_search_radius(self, max_search_radius: int):
        """Set the maximum search radius for fast cost updates."""
        assert max_search_radius > 0.0
        logger.info(f'Setting max XYZ search radius to: {max_search_radius}')
        self._lib.max_search_radius(self._engine, max_search_radius)

    @property
    def update_method(self):
        return self._bayesian_update_method

    @update_method.setter
    def update_method(self, method):
        """Set the method for updates, EXACT, APPROXIMATE, CUDA etc... """
        assert method in constants.BayesianUpdates
        logger.info(f'Setting Bayesian update method to: {method}')
        self._lib.set_update_mode(self._engine, method.value)
        self._bayesian_update_method = method

    @property
    def n_tracks(self):
        """Return the number of tracks found."""
        return self._lib.size(self._engine)

    @property
    def n_dummies(self):
        """Return the number of dummy objects (negative ID)."""
        return len(
            [d for d in itertools.chain.from_iterable(self.refs) if d < 0]
        )

    @property
    def tracks(self):
        """Return a sorted list of tracks, default is to sort by increasing
        length."""
        return [self[i] for i in range(self.n_tracks)]

    @property
    def refs(self):
        """Return tracks as a list of IDs (essentially pointers) to the original
        objects. Use this to write out HDF5 tracks. """
        tracks = []
        for i in range(self.n_tracks):
            # get the track length
            n = self._lib.track_length(self._engine, i)

            # set up some space for the output and  get the track data
            refs = np.zeros((n,), dtype='int32')
            _ = self._lib.get_refs(self._engine, refs, i)
            tracks.append(refs.tolist())

        return tracks

    @property
    def dummies(self):
        """Return a list of dummy objects."""
        return [
            self._lib.get_dummy(self._engine, -(i + 1))
            for i in range(self.n_dummies)
        ]

    @property
    def lbep(self):
        """Return an LBEP list describing the track lineage information.

        Notes
        -----
        L - a unique label of the track (label of markers, 16-bit positive)
        B - a zero-based temporal index of the frame in which the track begins
        E - a zero-based temporal index of the frame in which the track ends
        P - label of the parent track (0 is used when no parent is defined)
        R - label of the root track
        G - generational depth (from root)
        """

        def _lbep_table(t):
            return (
                t.ID,
                t.start,
                t.stop,
                t.parent,
                t.root,
                t.generation,
            )

        return [_lbep_table(t) for t in self.tracks]

    def _sort(self, tracks):
        """ Return a sorted list of tracks """
        return sorted(tracks, key=lambda t: len(t), reverse=True)

    @property
    def volume(self):
        """Return the imaging volume in the format xyzt. This is effectively
        the range of each dimension: [(xlo, xhi), ..., (zlo, zhi), (tlo, thi)].
        """
        vol = np.zeros((3, 2), dtype='float')
        self._lib.get_volume(self._engine, vol)
        return [tuple(vol[i, :].tolist()) for i in range(3)] + [
            self.frame_range
        ]

    @volume.setter
    def volume(self, volume: tuple):
        """Set the imaging volume.

        Parameters
        ----------
        volume : tuple
            A tuple describing the imaging volume.
        """
        if not isinstance(volume, tuple):
            raise TypeError('Volume must be a tuple')
        if len(volume) != 3 or any([len(v) != 2 for v in volume]):
            raise ValueError('Volume must contain three tuples (xyz)')
        self._lib.set_volume(self._engine, np.array(volume, dtype='float64'))
        logger.info(f'Set volume to {volume}')

    @property
    def motion_model(self):
        return self._motion_model

    @motion_model.setter
    def motion_model(self, new_model):
        """Set a new motion model. Must be of type MotionModel, either loaded
        from file or instantiating a MotionModel.

        Parameters
        ----------
        new_model : MotionModel
            A motion model to be used by the tracker.
        """

        if isinstance(new_model, btypes.MotionModel):
            # TODO(arl): model parsing for a user defined model
            model = new_model
        else:
            raise TypeError(
                'Motion model needs to be defined in /models/ or'
                'provided as a MotionModel object'
            )

        self._motion_model = model
        logger.info(f'Loading motion model: {model.name}')

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

    @property
    def object_model(self):
        return self._object_model

    @object_model.setter
    def object_model(self, new_model):
        """ Set a new object model. Must be of type ObjectModel, either loaded
        from file or instantiating an ObjectModel.

        Parameters
        ----------
        new_model : ObjectModel
        """

        if isinstance(new_model, btypes.ObjectModel):
            # this could be a user defined model
            # TODO(arl): model parsing
            model = new_model
        elif new_model is None:
            return
        else:
            raise TypeError(
                'Object model needs to be defined in /models/ or'
                'provided as a ObjectModel object'
            )

        self._object_model = model
        logger.info(f'Loading object model: {model.name}')

        # need to populate fields in the C++ library
        self._lib.model(
            self._engine,
            model.states,
            model.emission,
            model.transition,
            model.start,
        )

    @property
    def frame_range(self):
        return self._frame_range

    @frame_range.setter
    def frame_range(self, frame_range: tuple):
        if not isinstance(frame_range, tuple):
            raise TypeError('Frame range must be specified as a tuple')
        if frame_range[1] < frame_range[0]:
            raise ValueError('Frame range must be low->high')
        self._frame_range = frame_range

    @property
    def objects(self):
        return self._objects

    def append(self, objects):
        """Append a single track object, or list of objects to the stack. Note
        that the tracker will automatically order these by frame number, so the
        order here does not matter. This means several datasets can be
        concatenated easily, by running this a few times.

        Parameters
        ----------

        objects : list, np.ndarray
            A list of objects to track.
        """

        objects = localizations_to_objects(objects)

        for idx, obj in enumerate(objects):
            obj.ID = idx + len(self._objects)  # make sure ID tracks properly
            if not isinstance(obj, btypes.PyTrackObject):
                raise TypeError('track_object must be a PyTrackObject')

            self._frame_range[1] = max(obj.t, self._frame_range[1])
            _ = self._lib.append(self._engine, obj)

        # store a copy of the list of objects
        self._objects += objects

    def _stats(self, info_ptr):
        """ Cast the info pointer back to an object """

        if not isinstance(info_ptr, ctypes.POINTER(btypes.PyTrackingInfo)):
            raise TypeError('Stats requires the pointer to the object')

        return info_ptr.contents

    def track(self):
        """ Run the actual tracking algorithm """

        if not self._initialised:
            logger.error('Tracker has not been configured')
            return

        logger.info('Starting tracking... ')
        # ret, tm = timeit( lib.track,  self._engine )
        ret = self._lib.track(self._engine)

        # get the statistics
        stats = self._stats(ret)

        if not utils.log_error(stats.error):
            logger.info(
                (
                    f'SUCCESS. Found {self.n_tracks} tracks in'
                    f'{1+self._frame_range[1]} frames'
                )
            )

        # can log the statistics as well
        utils.log_stats(stats.to_dict())

    def track_interactive(self, step_size: int = 100):
        """Run the tracking in an interactive mode.

        Parameters
        ----------
        step_size : int, default=100
            The number of tracking steps to be taken before returning summary
            statistics. The tracking will be followed to completion, regardless
            of the step size provided.
        """

        # TODO(arl): this needs cleaning up to have some decent output
        if not self._initialised:
            logger.error('Tracker has not been configured')
            return

        logger.info('Starting tracking... ')

        stats = self.step()
        frm = 0

        # while not stats.complete and stats.error not in constants.ERRORS:
        while stats.tracker_active:
            logger.info(
                (
                    f'Tracking objects in frames {frm} to '
                    f'{min(frm+step_size-1, self._frame_range[1]+1)} '
                    f'(of {self._frame_range[1]+1})...'
                )
            )

            stats = self.step(step_size)
            utils.log_stats(stats.to_dict())
            frm += step_size

        if not utils.log_error(stats.error):
            logger.info('SUCCESS.')
            logger.info(
                (
                    f' - Found {self.n_tracks} tracks in '
                    f'{1+self._frame_range[1]} frames '
                    f'(in {stats.t_total_time}s)'
                )
            )
            logger.info(
                (
                    f' - Inserted {self.n_dummies} dummy objects to fill '
                    'tracking gaps'
                )
            )

    def step(self, n_steps: int = 1):
        """Run an iteration (or more) of the tracking. Mostly for interactive
        mode tracking."""
        if not self._initialised:
            return None
        return self._stats(self._lib.step(self._engine, n_steps))

    def hypotheses(self, params=None):
        """Calculate and return hypotheses using the hypothesis engine."""
        # raise NotImplementedError
        if not self.hypothesis_model:
            raise AttributeError('Hypothesis model has not been specified.')

        n_hypotheses = self._lib.create_hypotheses(
            self._engine,
            self.hypothesis_model,
            self.frame_range[0],
            self.frame_range[1],
        )

        # now get all of the hypotheses
        h = [
            self._lib.get_hypothesis(self._engine, h)
            for h in range(n_hypotheses)
        ]
        return h

    def optimize(self, **kwargs):
        return self.optimise(**kwargs)

    def optimise(self, options: dict = constants.GLPK_OPTIONS):
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

        logger.info(f'Loading hypothesis model: {self.hypothesis_model.name}')

        logger.info(
            f'Calculating hypotheses (relax: {self.hypothesis_model.relax})...'
        )
        hypotheses = self.hypotheses()

        # if we don't have any hypotheses return
        if not hypotheses:
            logger.warning('No hypotheses could be found.')
            return []

        # set up the track optimiser
        track_linker = optimiser.TrackOptimiser(options=options)
        track_linker.hypotheses = hypotheses
        selected_hypotheses = track_linker.optimise()
        optimised = [hypotheses[i] for i in selected_hypotheses]

        if not optimised:
            logger.warning('Optimization failed.')
            return []

        h_original = [h.type for h in hypotheses]
        h_optimise = [h.type for h in optimised]
        h_types = sorted(list(set(h_original)), key=lambda h: h.value)

        for h_type in h_types:
            logger.info(
                (
                    f' - {h_type}: {h_optimise.count(h_type)}'
                    f' (of {h_original.count(h_type)})'
                )
            )
        logger.info(f' - TOTAL: {len(hypotheses)} hypotheses')

        # now that we have generated the optimal sequence, merge all of the
        # tracks, delete fragments and assign divisions
        h_array = np.array(selected_hypotheses, dtype='uint32')
        h_array = h_array[np.newaxis, ...]
        self._lib.merge(self._engine, h_array, len(selected_hypotheses))
        logger.info(f'Completed optimization with {self.n_tracks} tracks')

        return optimised

    def __getitem__(self, idx: int):
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
        if nc > 0:
            c = children.tolist()
        else:
            c = []

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
        sz_cov = self.motion_model.measurements ** 2 + 1

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

    def export(self, filename: str, obj_type=None, filter_by=None):
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
        export_delegator(
            filename, self, obj_type=obj_type, filter_by=filter_by
        )

    def to_napari(self, ndim: int = 3, replace_nan: bool = True):
        """Return the data in a format for a napari tracks layer.
        See `utils.tracks_to_napari`."""
        return utils.tracks_to_napari(self.tracks, ndim, replace_nan=replace_nan)


if __name__ == "__main__":
    pass
