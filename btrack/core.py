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
from optimise import hypothesis
from optimise import linker

from datetime import datetime
from collections import OrderedDict
import itertools


# get the logger instance
logger = logging.getLogger('worker_process')

# if we don't have any handlers, set one up
if not logger.handlers:
    # configure stream handler
    log_formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')
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
print BTRACK_PATH











class PyTrackObject(ctypes.Structure):
    """ TrackObject

    Primitive class to store information about an object. Essentially a single
    object in a field of view, with some member variables to keep track of data
    associated with an object.

    Args:
        position: 2D/3D position
        dummy: is this a real object or a dummy object (e.g. when lost)
        label: object classification
        attributes:	object attributes, essentially metadata about object

    Properties:
        probability: class label probabilities

    Notes:
        Similar to the impy TrackObject class.

        TODO(arl): Add attributes and to/from JSON functions

    """

    _fields_ = [('ID', ctypes.c_uint),
                ('x', ctypes.c_double),
                ('y', ctypes.c_double),
                ('z', ctypes.c_double),
                ('t', ctypes.c_uint),
                ('dummy', ctypes.c_bool),
                ('states', ctypes.c_uint),
                ('label', ctypes.c_int),
                ('prob', ctypes.POINTER(ctypes.c_double))]

    def __init__(self):
        self.__raw_probability = None

    @property
    def probability(self):
        return self.__raw_probability
    @probability.setter
    def probability(self, probability):
        if not isinstance(probability, np.ndarray):
            raise TypeError('.probability should be a numpy array')
        self.__raw_probability = probability
        self.prob = probability.ctypes.data_as(ctypes.POINTER(ctypes.c_double))




class PyTrackingInfo(ctypes.Structure):
    """ PyTrackingStatistics

    Primitive class to store information about the tracking output.

    Params:
        error: error code from the tracker
        n_tracks: total number of tracks initialised during tracking
        n_active: number of active tracks
        n_conflicts: number of conflicts
        n_lost: number of lost tracks
        t_update_belief: time to update belief matrix in ms
        t_update_link: time to update links in ms
        t_total_time: total time to track objects
        p_link: typical probability of association
        p_lost: typical probability of losing track

    Notes:
        TODO(arl): should update to give more useful statistics, perhaps
        histogram of probabilities and timings.

    """

    _fields_ = [('error', ctypes.c_uint),
                ('n_tracks', ctypes.c_uint),
                ('n_active', ctypes.c_uint),
                ('n_conflicts', ctypes.c_uint),
                ('n_lost', ctypes.c_uint),
                ('t_update_belief', ctypes.c_float),
                ('t_update_link', ctypes.c_float),
                ('t_total_time', ctypes.c_float),
                ('p_link', ctypes.c_float),
                ('p_lost', ctypes.c_float),
                ('complete', ctypes.c_bool)]


    def to_dict(self):
        """ Return a dictionary of the statistics """
        # TODO(arl): make this more readable by converting seconds, milliseconds
        # and interpreting error messages?
        stats = {k:getattr(self, k) for k,typ in PyTrackingInfo._fields_}
        return stats








class MotionModel(object):
    """ MotionModel

    Kalman filter:
    'Is an algorithm which uses a series of measurements observed over time,
    containing noise (random variations) and other inaccuracies, and produces
    estimates of unknown variables that tend to be more precise than those that
    would be based on a single measurement alone.'

    Args:
        name: a name identifier
        measurements: the number of measurements of the system (typically x,y,z)
        states: the number of states of the system (typically >=measurements)
        A: State transition matrix
        B: Control matrix
        H: Observation matrix
        P: Initial covariance estimate
        Q: Estimated error in process
        R: Estimated error in measurements
        accuracy: integration limits for calculating the probabilities
        dt: time difference (always 1?)
        max_lost: number of frames without observation before marking as lost
        prob_not_assign: the default probability to not assign a track

    Members:
        reshape(): reshape matrices to the correct dimensions, will throw error
            if they are incorrectly sized.
        load(): load a motion model from a JSON file.

    Notes:
        This is just a wrapper for the data with a few convenience functions
        thrown in. Matrices must be stored Fortran style, because Eigen uses
        column major and Numpy uses row major storage.

    References:
        'A new approach to linear filtering and prediction problems.'
        Kalman RE, 1960 Journal of Basic Engineering
    """

    def __init__(self):
        self.name = 'Default'
        self.A = None
        self.H = None
        self.P = None
        self.G = None
        self.R = None
        self.measurements = None
        self.states = None
        self.dt = 1
        self.accuracy = 2.
        self.max_lost = constants.MAX_LOST
        self.prob_not_assign = constants.PROB_NOT_ASSIGN

    @property
    def Q(self):
        """ Return a Q matrix from the G matrix. """
        return self.G.transpose() * self.G

    def reshape(self):
        """ Reshapes matrices to the correct dimensions. Only need to call this
        if loading a model from a JSON file.

        Notes:
            Internally:
                Eigen::Matrix<double, m, s> H;
                Eigen::Matrix<double, s, s> Q;
                Eigen::Matrix<double, s, s> P;
                Eigen::Matrix<double, m, m> R;

        """
        s = self.states
        m = self.measurements

        # if we have defined a model, restructure matrices to the correct shapes
        # do some parsing to check that the model is specified correctly
        if s and m:
            shapes = {'A':(s,s), 'H':(m,s), 'P':(s,s), 'R':(m,m)}
            for m_name in shapes:
                try:
                    m_array = getattr(self, m_name)
                    r_matrix = np.reshape(m_array, shapes[m_name], order='C')
                except ValueError:
                    raise ValueError('Matrx {0:s} is incorrecly specified. '
                        '({1:d} entries for {2:d}x{3:d} matrix.)'.format(m_name,
                        len(m_array), shapes[m_name][0],
                        shapes[m_name][1]))

                setattr(self, m_name, r_matrix)
        else:
            raise ValueError('Cannot reshape matrices as MotionModel is '
                            'uninitialised')

    @staticmethod
    def load(filename):
        """ Load a model from file """
        return utils.read_motion_model(filename)









class ObjectModel(object):
    """ ObjectModel

    This is a class to deal with state transitions in the object, essentially
    a Hidden Markov Model.  Makes an assumption that the states are all
    observable, but with noise.

    Args:
        emission: the emission probability matrix
        transition: transition probabilities
        start: initial probabilities

    """
    def __init__(self):
        self.emission = None
        self.transition = None
        self.start = None
        self.states = None

    def reshape(self):
        """ Reshapes matrices to the correct dimensions. Only need to call this
        if loading a model from a JSON file.

        Notes:
            Internally:
                Eigen::Matrix<double, s, s> emission;
                Eigen::Matrix<double, s, s> transition;
            	Eigen::Matrix<double, s, 1> start;
        """
        if not self.states:
            raise ValueError('Cannot reshape matrices as ObjectModel is '
                            'uninitialised')
        s = self.states
        self.emission = np.reshape(self.emission, (s,s), order='C')
        self.transition = np.reshape(self.transition, (s,s), order='C')

    @staticmethod
    def load(filename):
        """ Load a model from file """
        return utils.read_object_model(filename)














class Tracklet(object):
    """ Tracklet

    Tracklet object for storing and updating linked lists of track objects.
    Forms the data structure for an individual tracklet.

    Args:
        ID: unique identifier
        data: trajectory
        kalman: Kalman filter output
        labels: class labels for each object
        fate: the fate of the track

    Members:
        __len__: length of the trajectory in frames (including interpolated)
        lost: whether the track was lost during tracking
        merge: a merging function to stitch together two tracks
        labeller: a mapping function from an integer to another label type (str)

    Properties:
        x: x position
        y: y position
        z: z position
        dummy: did this position arise from an actual measurement?
        parent:	parent tracklet
        root: root tracklet if a branching tree (ie cell division)
        motion_model: typically a reference to a Kalman filter or motion model

	Notes:
		TODO (arl) add the dummy field back, and the track merging. Also,
        clean up indexing into arrays.
    """

    def __init__(self,
                 ID,
                 data,
                 kalman=None,
                 labels=None,
                 parent=None,
                 fate=None):

        self.ID = ID
        self.__data = data
        self.__kalman = kalman
        self.__labels = labels
        self.__dummy = None

        self.root = None
        self.parent = parent
        self.type = None
        self.fate = fate

        # labeller is a function to convert an integer label to a string or
        # other type if required. default is to return the original integer
        self.labeller = utils.Labeller()

    def __len__(self):
        return self.__data.shape[0]

    @property
    def x(self): return self.__data[:,1].tolist()
    @property
    def y(self): return self.__data[:,2].tolist()
    @property
    def z(self): return self.__data[:,3].tolist()
    @property
    def t(self): return self.__data[:,0].tolist()
    @property
    def dummy(self): return self.__dummy

    @property
    def label(self):
        return [self.labeller(l) for l in self.__labels.tolist()]

    def kalman(self, index):
        """ Return the entire Kalman filter output for one parameter """
        #TODO(arl): get the kalman attribute by name
        return self.__kalman[:,index]
        #raise DeprecationWarning("Use mu() and covar() instead.")

    def mu(self, index):
        """ Return the Kalman filter mu. Note that we are only returning the mu
         for the positions (e.g. 3x1) """
        return np.matrix(self.__kalman[index,1:4]).reshape(3,1)

    def covar(self, index):
        """ Return the Kalman filter covariance matrix. Note that we are
        only returning the covariance matrix for the positions (e.g. 3x3) """
        return np.matrix(self.__kalman[index,4:13]).reshape(3,3)

    def predicted(self, index):
        """ Return the motion model prediction for the given timestep. """
        return np.matrix(self.__kalman[index,13:]).reshape(3,1)

    def to_dict(self):
        """ Return a dictionary of the tracklet which can be used for JSON
        export. This is an ordered dictionary for nicer JSON output.
        """
        # TODO(arl): add the Kalman filter output here too
        trk_tuple = (('ID',self.ID), ('length',len(self)), ('root',self.root),
            ('parent',self.parent), ('x',self.x),( 'y',self.y), ('z',self.z),
            ('t',self.t),('label',self.label),('fate',self.fate))

        return OrderedDict( trk_tuple )

    def to_array(self):
        """ Return a numpy array of the tracklet which can be used for MATLAB
        export. """
        # TODO(arl): add the Kalman filter output here too
        return np.hstack((self.__data, np.ones((len(self),1))*self.ID))

    def in_frame(self, frame):
        """ Return true or false as to whether the track is in the frame """
        return self.__data[0,0]<=frame and self.__data[-1,0]>=frame

    def trim(self, frame, tail=75):
        """ Trim the tracklet and return one with the trimmed data """
        d = self.__data.copy()
        idx = [self.t.index(t) for t in self.t if t<=frame and t>=frame-tail]
        d = d[idx,:]
        return Tracklet(self.ID, d)










def timeit(func, *args):
    """ Temporary function. Will remove in final release """
    t_start = time.time()
    ret = func(*args)
    t_elapsed = time.time() - t_start
    return ret, t_elapsed

def np_dbl_p():
    """ Temporary function. Will remove in final release """
    return np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS')

def np_dbl_pc():
    """ Temporary function. Will remove in final release """
    return np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='F_CONTIGUOUS')

def np_uint_p():
    """ Temporary function. Will remove in final release """
    return np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2, flags='C_CONTIGUOUS')

def np_int_p():
    """ Temporary function. Will remove in final release """
    return np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')







class LibraryWrapper(object):
    """ LibraryWrapper

    This is a container and interface class to the btrack library. This can
    be shared between the tracker and the optimiser to provide a uniform
    interface.

    """

    lib = utils.load_library(os.path.join(BTRACK_PATH,'libs/libtracker'))

    # deal with constructors/destructors
    lib.new_interface.restype = ctypes.c_void_p
    lib.new_interface.argtypes = [ctypes.c_bool]

    lib.del_interface.restype = None
    lib.del_interface.argtypes = [ctypes.c_void_p]

    # set the motion model
    lib.motion.restype = None
    lib.motion.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
                            np_dbl_p(), np_dbl_p(), np_dbl_p(), np_dbl_p(),
                            np_dbl_p(), ctypes.c_double, ctypes.c_double,
                            ctypes.c_uint, ctypes.c_double]

    # set the object model
    # lib.model.restype = None
    # lib.model.argtypes = [ctypes.c_void_p, ctypes.c_uint, np_dbl_p(),
    #                         np_dbl_p(), np_dbl_p()]

    # append a new observation
    lib.append.restype = None
    lib.append.argtypes = [ctypes.c_void_p, PyTrackObject]

    # run the complete tracking
    lib.track.restype = ctypes.POINTER(PyTrackingInfo)
    lib.track.argtypes = [ctypes.c_void_p]

    # run one or more steps of the tracking, interactive mode
    lib.step.restype = ctypes.POINTER(PyTrackingInfo)
    lib.step.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    # get an individual track length
    lib.track_length.restype = ctypes.c_uint
    lib.track_length.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    # get a track
    lib.get.restype = ctypes.c_uint
    lib.get.argtypes = [ctypes.c_void_p, np_dbl_p(), ctypes.c_uint]

    # get a track, by reference
    lib.get_refs.restype = ctypes.c_uint
    lib.get_refs.argtypes = [ctypes.c_void_p, np_int_p(), ctypes.c_uint]

    # get the parent ID (i.e. pre-division)
    lib.get_parent.restype = ctypes.c_uint
    lib.get_parent.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    # get the fate of the track
    lib.get_fate.restype = ctypes.c_uint
    lib.get_fate.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    # get the kalman filtered position
    lib.get_kalman_mu.restype = ctypes.c_uint
    lib.get_kalman_mu.argtypes = [ctypes.c_void_p, np_dbl_p(), ctypes.c_uint]

    # get the kalman covariance
    lib.get_kalman_covar.restype = ctypes.c_uint
    lib.get_kalman_covar.argtypes = [ctypes.c_void_p, np_dbl_p(), ctypes.c_uint]

    # get the predicted position at each time step
    lib.get_kalman_pred.restype = ctypes.c_uint
    lib.get_kalman_pred.argtypes = [ctypes.c_void_p, np_dbl_p(), ctypes.c_uint]

    # get the label of the object
    lib.get_label.restype = ctypes.c_uint
    lib.get_label.argtypes = [ctypes.c_void_p, np_uint_p(), ctypes.c_uint]

    # get the imaging volume
    lib.get_volume.restype = None
    lib.get_volume.argtypes = [ctypes.c_void_p, np_dbl_p()]

    # return a dummy object by reference
    lib.get_dummy.restype = PyTrackObject
    lib.get_dummy.argtypes = [ctypes.c_void_p, ctypes.c_int]

    # get the number of tracks
    lib.size.restype = ctypes.c_uint
    lib.size.argtypes = [ctypes.c_void_p]

    # calculate the hypotheses
    lib.create_hypotheses.restype = ctypes.c_uint
    lib.create_hypotheses.argtypes = [ctypes.c_void_p,
        hypothesis.PyHypothesisParams,
        ctypes.c_uint, ctypes.c_uint]

    # get a hypothesis by ID
    lib.get_hypothesis.restype = hypothesis.Hypothesis
    lib.get_hypothesis.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    # merge following optimisation
    lib.merge.restype = None
    lib.merge.argtypes = [ctypes.c_void_p, np_uint_p(), ctypes.c_uint]




# get a reference to the library
lib = LibraryWrapper.lib




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

            p_file = os.path.join(BTRACK_PATH,'models/', config['HypothesisModel'])
            params = hypothesis.PyHypothesisParams.load( p_file )
            logger.info('Loading hypothesis model: {0:s}'.format(params.name))
            self.hypothesis_model = params

        self.__initialised = True


    def __len__(self): return self.n_tracks

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
            model_fn = os.path.join(BTRACK_PATH,'models/',new_model)
            model = utils.read_motion_model(model_fn)
        elif isinstance(new_model, MotionModel):
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
            model_fn = os.path.join(BTRACK_PATH,'models/',new_model)
            model = utils.read_object_model(model_fn)
        elif isinstance(new_model, ObjectModel):
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
            if not isinstance(obj, PyTrackObject):
                raise TypeError('track_object must be a PyTrackObject')

            self.__frame_range[1] = max(obj.t, self.__frame_range[1])
            ret = lib.append( self.__engine, obj )


    def xyzt(self, array):
        """ Pass in a numpy array of data """
        raise NotImplementedError

    def __stats(self, info_ptr):
        """ Cast the info pointer back to an object """

        if not isinstance(info_ptr, ctypes.POINTER(PyTrackingInfo)):
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
        while not stats.complete and stats.error == 910:
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
            logger.info(' - Inserted {0:d} dummy objects to in-fill '
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
        logger.info(' - Found {0:d} hypotheses'.format(len(hypotheses)))

        # set up the track optimiser
        track_linker = linker.TrackOptimiser()
        track_linker.hypotheses = hypotheses
        selected_hypotheses = track_linker.optimise()

        # now that we have generated the optimal sequence, merge all of the
        # tracks, delete fragments and assign divisions
        h_array = np.array(selected_hypotheses, dtype='uint32')
        h_array = h_array[np.newaxis,...]
        lib.merge(self.__engine, h_array, len(selected_hypotheses))

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

        # get the track data
        _ = lib.get(self.__engine, trk, index)
        _ = lib.get_label(self.__engine, lbl, index)
        p = lib.get_parent(self.__engine, index)

        # optional, we can grab the motion model data too
        if not self.return_kalman:
            return Tracklet(index, trk, labels=lbl[:,1], parent=p)

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
        return Tracklet(index, trk, kalman=kal, labels=lbl[:,1], parent=p)


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



    def cleanup(self):
        """ Clean up following tracking. Can be used to remove static objects
        and links that are greater than the maximum distance permitted """
        raise NotImplementedError









if __name__ == "__main__":
    pass
