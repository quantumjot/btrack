#!/usr/bin/env python
#-------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------


__author__ = "Alan R. Lowe"
__email__ = "a.lowe@ucl.ac.uk"

import os
import json
import time
import ctypes
import logging
import numpy as np

import utils
import constants
import btypes

from optimise import hypothesis, optimiser

from datetime import datetime
from collections import OrderedDict
import itertools


# get the logger instance
logger = logging.getLogger('worker_process')

# if we don't have any handlers, set one up
if not logger.handlers:
    # configure stream handler
    log_formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)


logger.info('btrack (v{0:s}) library imported'.format(constants.__version__))


def log_to_file(pth, level=None):
    """ Set the logging to output to a directory """
    raise NotImplementedError




# TODO(arl): sort this out with final packaging!
BTRACK_PATH = os.path.dirname(os.path.abspath(__file__))

























def timeit(func, *args):
    """ Temporary function. Will remove in final release """
    t_start = time.time()
    ret = func(*args)
    t_elapsed = time.time() - t_start
    return ret, t_elapsed











# get a reference to the library
import libwrapper
lib = libwrapper.LibraryWrapper.lib

class BayesianTracker(object):
    """ BayesianTracker

    BayesianTracker is a multi object tracking algorithm, specifically
    used to reconstruct tracks in crowded fields. Here we use a probabilistic
    network of information to perform the trajectory linking. This method uses
    positional information (position, velocity ...) as well as visual
    information (labels, features...) for track linking.

    The tracking algorithm assembles reliable sections of track that do not
    contain splitting events (tracklets). Each new tracklet initiates a
    probabilistic model in the form of a Kalman filter (Kalman, 1960), and
    utilises this to predict future states (and error in states) of each of the
    objects in the field of view.  We assign new observations to the growing
    tracklets (linking) by evaluating the posterior probability of each
    potential linkage from a Bayesian belief matrix for all possible linkages
    (Narayana and Haverkamp, 2007). The best linkages are those with the highest
    posterior probability.

    This class is a wrapper for the C++ implementation of the BayesianTracker.

    Data can be passed in in the following formats:
        - impy TrackObject
        - Optional JSON files using loaders
        - HDF

    Can be used with ContextManager support, like this:

        >>> with BayesianTracker() as tracker:
        >>>    tracker.append(observations)
        >>>    tracker.track()

    The tracker can be used to return all of the original data neatly packaged
    into tracklet objects, or as a nested list of references to the original
    data sets. The latter is useful if this is only the first part of a tracking
    protocol, or other metadata is needed for further analysis. The references
    can be used to make symbolic links in HDF5 files, for example.

    Use the .tracks to return Tracklets, or .refs to return the references.

    Use optimise to generate hypotheses for global optimisation. Read the
    TrackLinker documentation for more information about the track linker.

    Members:
        append(): append an object (or list of objects)
        xyzt(): set an entire array of data
        track(): run the tracking algorithm
        track_interactive(): run the tracking in interactive mode
        export(): export the data to a JSON format
        cleanup(): clean up the tracks according to some metrics
        optimise(): run the optimiser

    Args:
        motion_model: a motion model to make motion predictions
        object_model: an object model to make state predictions
        return_kalman: boolean to request the Kalman debug info

    Properties:
        n_tracks: number of found tracks
        tracks: the tracks themselves
        refs: the tracks (by refernce)
        volume: the imaging volume [x,y,z,t]
        frame_range: the frame range for tracking, essentially the last
            dimension of volume
        max_search_radius: maximum search radius when using fast cost update

    Notes:
        TODO(arl): lower precision for tracking output?
        DONE(arl): return list of references - 31-10-2017
        DONE(arl): clean up setting up of models - 27-12-2017

    References:
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

    def __init__(self, verbose=True):
        """ Initialise the BayesianTracker C++ engine and parameters """

        # default parameters
        self.__motion_model = None
        self.__object_model = None
        self.__frame_range = [0,0]
        self.__max_search_radius = 100.0
        self.return_kalman = False

        # do not initialise until the init() has been run
        self.__initialised = False

        # get an instance of the engine
        self.__engine = lib.new_interface(verbose)


    def __enter__(self):
        logger.info('Starting BayesianTracker session')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info('Ending BayesianTracker session')
        lib.del_interface( self.__engine )


    def configure_from_file(self, filename):
        """ Configure the tracker from a configuration file """

        if not filename.endswith('.json'):
            filename+='.json'

        with open(os.path.join(BTRACK_PATH,'models/',filename), 'r') as cfg_file:
            config = json.load(cfg_file)

        if 'TrackerConfig' not in config:
            raise AttributeError('Configuration file is incorrectly specified')

        # configure the tracker from the file
        self.configure( config['TrackerConfig'] )


    def configure(self, config):
        """ Configure the tracker with a motion model, an object model and
        hypothesis generation_parameters. The configuration should be
        specified as follows:

            config = {'MotionModel':'cell_motion.json',
                      'ObjectModel':'cell_object.json',
                      'HypothesisModel':'cell_hypothesis.json'}

        """

        if not isinstance(config, dict):
            raise TypeError('configuration must be a dictionary')

        if 'MotionModel' in config:
            self.motion_model = config['MotionModel']

        if 'ObjectModel' in config:
            self.object_model = config['ObjectModel']

        if 'HypothesisModel' in config:
            # set up hypothesis model
            p_file = os.path.join(BTRACK_PATH,'models',config['HypothesisModel'])
            params = hypothesis.PyHypothesisParams.load( p_file )
            logger.info('Loading hypothesis model: {0:s}'.format(params.name))
            self.hypothesis_model = params

        # set the maximum search radius
        self.max_search_radius = 100.0

        self.__initialised = True


    def __len__(self): return self.n_tracks

    @property
    def max_search_radius(self):
        return self.__max_search_radius
    @max_search_radius.setter
    def max_search_radius(self, max_search_radius):
        """ Set the maximum search radius for fast cost updates """
        assert(max_search_radius>0. and max_search_radius<=100.)
        logger.info('Setting maximum XYZ search radius to {0:2.2f}...'
                    .format(max_search_radius))
        lib.max_search_radius(self.__engine, max_search_radius)



    @property
    def n_tracks(self):
        """ Return the number of tracks found """
        return lib.size( self.__engine )
    @property
    def n_dummies(self):
        """ Return the number of dummy objects (negative ID) """
        return len([d for d in itertools.chain.from_iterable(self.refs) if d<0])

    @property
    def tracks(self):
        """ Return a sorted list of tracks, default is to sort by increasing
        length """
        return self.__sort( [self[i] for i in xrange(self.n_tracks)] )

    @property
    def refs(self):
        """ Return tracks as a list of IDs (essentially pointers) to the
        original objects. Use this to write out HDF5 tracks. """
        tracks = []
        for i in xrange(self.n_tracks):
            # get the track length
            n = lib.track_length(self.__engine, i)

            # set up some space for the output and  get the track data
            refs = np.zeros((1,n),dtype='int32')
            _ = lib.get_refs(self.__engine, refs, i)
            tracks.append(refs.tolist()[0])

        return self.__sort(tracks)


    def __sort(self, tracks):
        """ Return a sorted list of tracks """
        return sorted(tracks, key=lambda t:len(t), reverse=True)

    @property
    def volume(self):
        """ Return the imaging volume in the format xyzt. This is effectively
        the range of each dimension: [(xlo,xhi),...,(zlo,zhi),(tlo,thi)]
        """
        vol = np.zeros((3,2),dtype='float')
        lib.get_volume(self.__engine, vol)
        return [tuple(vol[i,:].tolist()) for i in xrange(3)]+[self.frame_range]

    @property
    def motion_model(self):
        return self.__motion_model
    @motion_model.setter
    def motion_model(self, new_model):
        """ Set a new motion model. Must be of type MotionModel, either loaded
        from file or instantiating a MotionModel.

        Args:
            new_model: can be a string or a user defined MotionModel class

        Raises:
            TypeError is cannot determine the type of motion model
        """

        if isinstance(new_model, basestring):
            # load from the models directory
            model_fn = os.path.join(BTRACK_PATH,'models',new_model)
            model = utils.read_motion_model(model_fn)
        elif isinstance(new_model, btypes.MotionModel):
            # this could be a user defined model
            # TODO(arl): model parsing
            model = new_model
        else:
            raise TypeError('Motion model needs to be defined in /models/ or'
            'provided as a MotionModel object')

        self.__motion_model = model
        logger.info('Loading motion model: {0:s}'.format(model.name))

        # need to populate fields in the C++ library
        lib.motion( self.__engine,
            model.measurements, model.states, model.A,
            model.H, model.P, model.Q, model.R, model.dt, model.accuracy,
            model.max_lost, model.prob_not_assign )


    @property
    def object_model(self):
        return self.__object_model
    @object_model.setter
    def object_model(self, new_model):
        """
        Set a new object model. Must be of type ObjectModel, either loaded
        from file or instantiating an ObjectModel.

        Args:
            new_model: can be a string or a user defined ObjectModel class

        Raises:
            TypeError if cannot determine the type of motion model
        """

        if isinstance(new_model, basestring):
            if not new_model: return
            # load from the models directory
            model_fn = os.path.join(BTRACK_PATH,'models',new_model)
            model = utils.read_object_model(model_fn)
        elif isinstance(new_model, btypes.ObjectModel):
            # this could be a user defined model
            # TODO(arl): model parsing
            model = new_model
        else:
            raise TypeError('Object model needs to be defined in /models/ or'
            'provided as a ObjectModel object')

        if model is None: return

        self.__object_model = model
        logger.info('Loading object model: {0:s}'.format(model.name))

        # need to populate fields in the C++ library
        lib.model( self.__engine, model.states, model.emission,
            model.transition, model.start )




    @property
    def frame_range(self):
        return self.__frame_range
    @frame_range.setter
    def frame_range(self, frame_range):
        if not isinstance(frame_range, tuple):
            raise TypeError('Frame range must be specified as a tuple')
        if frame_range[1] < frame_range[0]:
            raise ValueError('Frame range must be low->high')
        self.__frame_range = frame_range




    def append(self, objects):
        """ Append a single track object, or list of objects to the stack. Note
        that the tracker will automatically order these by frame number, so the
        order here does not matter. This means several datasets can be
        concatenated easily, by running this a few times. """

        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            if not isinstance(obj, btypes.PyTrackObject):
                raise TypeError('track_object must be a PyTrackObject')

            self.__frame_range[1] = max(obj.t, self.__frame_range[1])
            ret = lib.append( self.__engine, obj )


    def xyzt(self, array):
        """ Pass in a numpy array of data """
        raise NotImplementedError

    def __stats(self, info_ptr):
        """ Cast the info pointer back to an object """

        if not isinstance(info_ptr, ctypes.POINTER(btypes.PyTrackingInfo)):
            raise TypeError('Stats requires the pointer to the object')

        return info_ptr.contents



    def track(self):
        """ Run the actual tracking algorithm """

        if not self.__initialised:
            logger.info('Using default parameters')
            self.configure({'MotionModel':'constant_velocity.json'})

        logger.info('Starting tracking... ')
        ret, tm = timeit( lib.track,  self.__engine )

        # get the statistics
        stats = self.__stats(ret)

        if not utils.log_error(stats.error):
            logger.info('SUCCESS. Found {0:d} tracks in {1:d} frames (in '
                '{2:.2f}s)'.format(self.n_tracks, 1+self.__frame_range[1],
                tm))

        # can log the statistics as well
        utils.log_stats(stats.to_dict())

    def track_interactive(self, step_size=100):
        """ Run the tracking in an interactive mode """

        # TODO(arl): this needs cleaning up to have some decent output
        if not self.__initialised:
            logger.info('Using default parameters')
            self.configure({'MotionModel':'constant_velocity.json'})

        logger.info('Starting tracking... ')

        stats = self.step()
        frm = 0
        # while not stats.complete and stats.error == 910:
        while not stats.complete and stats.error not in constants.ERRORS:
            logger.info('Tracking objects in frames {0:d} to '
                '{1:d} (of {2:d})...'.format(frm, min(frm+step_size-1,
                self.__frame_range[1]+1), self.__frame_range[1]+1))

            stats = self.step(step_size)
            utils.log_stats(stats.to_dict())
            frm+=step_size

        if not utils.log_error(stats.error):
            logger.info('SUCCESS.')
            logger.info(' - Found {0:d} tracks in {1:d} frames (in '
                '{2:.2f}s)'.format(self.n_tracks, 1+self.__frame_range[1],
                stats.t_total_time))
            logger.info(' - Inserted {0:d} dummy objects to fill '
                'tracking gaps'.format(self.n_dummies))



    def step(self, n_steps=1):
        """ Run an iteration (or more) of the tracking. Mostly for
        interactive mode tracking """
        if not self.__initialised: return None
        return self.__stats(lib.step( self.__engine, n_steps ))

    def hypotheses(self, params=None):
        """ Calculate and return hypotheses using the hypothesis engine """
        # raise NotImplementedError
        if not self.hypothesis_model:
            raise AttributeError('Hypothesis model has not been specified.')

        n_hypotheses = lib.create_hypotheses(self.__engine,
            self.hypothesis_model, self.frame_range[0], self.frame_range[1])

        # now get all of the hypotheses
        h = [lib.get_hypothesis(self.__engine, h) for h in xrange(n_hypotheses)]
        return h


    def optimise(self):
        """ Optimise the tracks. This generates the hypotheses for track merges,
        branching etc, runs the optimiser and then performs track merging,
        removal of track fragments, renumbering and assignment of branches.

        TODO(arl): need to check whether optimiser parameters have been
        specified
        """
        logger.info('Calculating hypotheses from tracklets...')
        hypotheses = self.hypotheses()

        # set up the track optimiser
        track_linker = optimiser.TrackOptimiser()
        track_linker.hypotheses = hypotheses
        selected_hypotheses = track_linker.optimise()
        optimised = [hypotheses[i] for i in selected_hypotheses]

        h_original = [h.type for h in hypotheses]
        h_optimise = [h.type for h in optimised]

        for h_type in set(h_original):
            logger.info(' - {0:s}: {1:d} (of {2:d})'.format(h_type,
                        h_optimise.count(h_type), h_original.count(h_type)))
        logger.info(' - TOTAL: {0:d} hypotheses'.format(len(hypotheses)))

        # now that we have generated the optimal sequence, merge all of the
        # tracks, delete fragments and assign divisions
        h_array = np.array(selected_hypotheses, dtype='uint32')
        h_array = h_array[np.newaxis,...]
        lib.merge(self.__engine, h_array, len(selected_hypotheses))

        return optimised

    def __getitem__(self, index):
        """ Grab a track from the BayesianTracker object.

        TODO (arl): This needs cleaning up. Also, we should just return a
        reference to the original object as a list rather than duplicating,
        which is currently memory intensive. Also 64-bit precision is
        probably overkill.
        """

        # get the size of the Kalman arrays
        sz_mu = self.motion_model.measurements + 1
        sz_cov = self.motion_model.measurements**2 + 1

        # get the track length
        n = lib.track_length(self.__engine, index)

        # set up some space for the output (could store as 32-bit, do we need)
        trk = np.zeros((n, sz_mu),dtype='float')
        lbl = np.zeros((n, 2),dtype='uint32')
        children = np.zeros((2,1),dtype='int32')

        # get the track data
        _ = lib.get(self.__engine, trk, index)
        _ = lib.get_label(self.__engine, lbl, index)
        _ = lib.get_children(self.__engine, children, index)
        p = lib.get_parent(self.__engine, index)
        f = lib.get_fate(self.__engine, index)

        # convert the array of children to a python list
        # TODO(arl): this is super lazy, should just make a vector for c
        c = np.squeeze(children).tolist()

        # and remove any without children: [0,0]
        if all([i==0 for i in c]): c = []

        # optional, we can grab the motion model data too
        if not self.return_kalman:
            return btypes.Tracklet(index,
                                   trk,
                                   labels=lbl[:,1],
                                   parent=p,
                                   children=c,
                                   fate=f)

        # otherwise grab the kalman filter data
        kal_mu = np.zeros((n, sz_mu),dtype='float')     # kalman filtered
        kal_cov = np.zeros((n, sz_cov),dtype='float')   # kalman covariance
        kal_pred = np.zeros((n, sz_mu),dtype='float')   # motion model predict

        n_kal = lib.get_kalman_mu(self.__engine, kal_mu, index)
        _ = lib.get_kalman_covar(self.__engine, kal_cov, index)
        _ = lib.get_kalman_pred(self.__engine, kal_pred, index)

        # cat the data [mu(0),...,mu(n),cov(0,0),...cov(n,n), pred(0),..]
        kal = np.hstack((kal_mu, kal_cov[:,1:], kal_pred[:,1:]))

        # return the tracklet
        # TODO(arl) make sure this is the correct ID, since we may have lost
        # some during the track merging/optimisation phase

        # make a new track object and return it
        trk = btypes.Tracklet(index,
                              trk,
                              kalman=kal,
                              labels=lbl[:,1],
                              parent=p,
                              children=c,
                              fate=f)

        return trk


    def get_dummy(self, dummy_idx):
        """ Return a PyTrackObject for a dummy object. This is useful if we
        are only returning references from the tracking, since we need to
        construct new objects for dummy objects inserted in to tracks.
        The user shouldn't really need this function, it should only be used
        when exporting to HDF5 files, since we need to make a new dummy group.
        """
        return lib.get_dummy(self.__engine, dummy_idx)

    def get_fate(self, index):
        """ Return the fate of the track. The fates can be used to sort tracks
        by type, for example by tracks that terminate in division or apoptosis.
        """
        return lib.get_fate(self.__engine, index)

    @property
    def dummies(self):
        """ Return all of the dummies """
        d_idx = [d for d in itertools.chain.from_iterable(self.refs) if d<0]
        d_idx = sorted(d_idx, reverse=True)
        return [self.get_dummy(idx) for idx in d_idx]

    def export(self, filename):
        """ Export the track data in the appropriate format for subsequent
        analysis. Note that the HDF5 format is intended to store references to
        objects, rather than the tracks themselves, so we need to deal with
        that differently here...

        TODO(arl): Make sure that we are working with an exisiting HDF5 file!
        """
        # log the output
        logger.info('Exporting {0:d} tracks to file...'.format(self.n_tracks))
        if not filename.endswith('hdf5'):
            utils.export(filename, self.tracks)
        else:
            utils.export_HDF(filename, self.refs, dummies=self.dummies)



    def cleanup(self, sigma=2.5):
        """ Clean up following tracking. Can be used to remove static objects
        and links that are greater than the maximum distance permitted """
        dynamic_track = lambda trk: (np.std(trk.x)+np.std(trk.y))*0.5 > sigma
        return [t for t in self.tracks if len(t)>1 and dynamic_track(t)]
        #raise NotImplementedError









if __name__ == "__main__":
    pass
