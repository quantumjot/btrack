#!/usr/bin/env python

#-------------------------------------------------------------------------------
# Name:     BayesianTracker
# Purpose:  A multi object tracking library, specifically used to reconstruct
#           tracks in crowded fields. Here we use a probabilistic network of
#           information to perform the trajectory linking. This method uses
#           positional and visual information for track linking.
#
# Authors:	Alan R. Lowe (arl) a.lowe@ucl.ac.uk
#
# License:  See LICENSE.md
#
# Created:	14/08/2014
#-------------------------------------------------------------------------------


__author__ = "Alan R. Lowe"
__email__ = "code@arlowe.co.uk"

import re
import os
import numpy as np
import time
import platform
import ctypes

import h5py
import csv
import json

import logging

import core
import constants

from collections import OrderedDict




class TrackingError(Exception):
    """ Raise an error from the BayesianTracker """




class Labeller(object):
    """ Labeller

    A class to enable coding and decoding of labels from classification.

    Args:
        labels: a tuple of labels

    Methods:
        decode: return an integer representation of the label
        encode: return the string? representation of the label

    Notes:
        None
    """
    def __init__(self, labels=()):
        self.labels = labels

    @property
    def labels(self):
        return self.__labels
    @labels.setter
    def labels(self, labels):
        if not isinstance(labels, tuple):
            raise TypeError('Labeller requires a tuple of labels')

        # make sure that we don't have duplicates, but retain order
        labels = list(OrderedDict((x, True) for x in labels).keys())
        self.__labels = labels

    def __call__(self, label): return self.decode(label)

    def decode(self, label):
        """ Return an index to a label """
        if not self.labels: return label
        return self.labels.index(label)

    def encode(self, label):
        if not self.labels: return label
        return self.labels[index]










def load_library(filename):
    """ Return the platform for shared library loading.  Take care of loading
    the appropriate shared library.

    Args:
        filename: filename for the library

    Raises:
        logging warning if windows is used
    """

    if not isinstance(filename, basestring):
        raise TypeError('Filename must be a string')

    lib_file, ext = os.path.splitext(filename)

    system = platform.system()
    version = platform.version()
    release = platform.release()

    if system is 'Windows':
        logging.warning('Windows is not fully supported yet. libtracker.DLL '
                        'must be compiled.')

    file_ext = {'Linux':'.so', 'Darwin':'.dylib', 'Windows':'.DLL'}

    full_lib_file = lib_file + file_ext[system]

    try:
        lib = ctypes.cdll.LoadLibrary(full_lib_file)
        logging.info('Loaded btrack: {0:s}'.format(full_lib_file))
    except IOError:
        raise IOError('Cannot load shared library {0:s}'.format(full_lib_file))

    return lib










def log_error(err_code):
    """ Take an error code from the tracker and log an error for the user.

    #define SUCCESS 900
    #define ERROR_empty_queue 901
    #define ERROR_no_tracks 902
    #define ERROR_no_useable_frames 903

    """

    if err_code in constants.ERRORS:
        logging.error('ERROR: {0:s}'.format(constants.ERRORS[err_code]))
        return True

    return False



def log_stats(stats):
    """ Take the statistics from the track and log the output """

    if log_error(stats['error']): return

    logging.info(' - Timing (Bayesian updates: {0:.2f}ms, Linking:'
                ' {1:.2f}ms)'.format(stats['t_update_belief'],
                stats['t_update_link']))

    logging.info(' - Probabilities (Link: {0:.5f}, Lost:'
                ' {1:.5f})'.format(stats['p_link'], stats['p_lost']))

    if stats['complete']:
        return

    logging.info(' - Stats (Active: {0:d}, Lost: {1:d}, Conflicts '
                'resolved: {2:d})'.format(stats['n_active'],
                stats['n_lost'], stats['n_conflicts']))








def read_motion_model(filename):
    """ read_motion_model

    Read in a motion model description file and return a dictionary containing
    the appropriate parameters.

    Motion models can be described using JSON format, with a basic structure
    as follows:

        {
          "MotionModel":{
            "name": "ConstantVelocity",
            "dt": 1.0,
            "measurements": 3,
            "states": 6,
            "accuracy": 2.0,
            "A": {
              "matrix": [1,0,0,1,0,0,...
              ...
              ] }
            }
        }

    Matrices are flattened JSON arrays.

    Most are self explanatory, except accuracy (perhaps a misnoma) - this
    represents the integration limits when determining the probabilities from
    the multivariate normal distribution.

    Args:
        filename: a JSON file describing the motion model.

    Returns:
        model: a core.MotionModel instance for passing to BayesianTracker

    Notes:
        Note that the matrices are stored as 1D matrices here. In the future,
        this could form part of a Python only motion model.

        TODO(arl): More parsing of the data/reshaping arrays. Raise an
        appropriate error if there is something wrong with the model definition.
    """

    matrices = frozenset(['A','H','P','G','R'])

    model = core.MotionModel()

    with open(filename, 'r') as j:
        modelfile = json.load(j)

        if 'MotionModel' not in modelfile.keys():
            raise ValueError('Not a valid motion model file')

        m = modelfile['MotionModel']

        # set some standard params
        model.name = m['name'].encode('utf-8')
        model.dt = m['dt']
        model.measurements = m['measurements']
        model.states = m['states']
        model.accuracy = m['accuracy']
        model.prob_not_assign = m['prob_not_assign']
        model.max_lost = m['max_lost']

        for matrix in matrices:
            if 'sigma' in m[matrix]:
                sigma = m[matrix]['sigma']
            else:
                sigma = 1.0
            m_data = np.matrix(m[matrix]['matrix'],dtype='float')
            setattr(model, matrix, m_data*sigma)

    # call the reshape function to set the matrices to the correct shapes
    model.reshape()

    return model









def read_object_model(filename):
    """ read_object_model

    Read in a object model description file and return a dictionary containing
    the appropriate parameters.

    Object models can be described using JSON format, with a basic structure
    as follows:

        {
          "ObjectModel":{
            "name": "UniformState",
            "states": 1,
            "transition": {
              "matrix": [1] }
              ...
            }
        }

    Matrices are flattened JSON arrays.

    Args:
        filename: a JSON file describing the object model.

    Returns:
        model: a core.ObjectModel instance for passing to BayesianTracker

    Notes:
        Note that the matrices are stored as 1D matrices here. In the future,
        this could form part of a Python only object model.

        TODO(arl): More parsing of the data/reshaping arrays. Raise an
        appropriate error if there is something wrong with the model definition.
    """

    matrices = frozenset(['transition','emission','start'])

    model = core.ObjectModel()

    with open(filename, 'r') as j:
        modelfile = json.load(j)

        if 'ObjectModel' not in modelfile.keys():
            raise ValueError('Not a valid object model file')

        m = modelfile['ObjectModel']

        # set some standard params
        model.name = m['name'].encode('utf-8')
        model.states = m['states']

        for matrix in matrices:
            m_data = np.matrix(m[matrix]['matrix'],dtype='float')
            setattr(model, matrix, m_data)

    # call the reshape function to set the matrices to the correct shapes
    model.reshape()

    return model




def crop_volume(objects, volume=constants.VOLUME):
    """ Return a list of objects that fall within a certain volume """
    axes = ['x','y','z']
    within = lambda o: all([getattr(o, a)>=v[0] and getattr(o, a)<=v[1] for a,v in zip(axes, volume)])
    return [o for o in objects if within(o)]








def export(filename, tracks):
    """ export

    Generic exporter of track data. Infers file type from extension and writes
    appropriate file type.

    Args:
        filename: full path to output file. If no extension is specified, use
            JSON by default.
        tracks: a list of Tracklet objects to write out.

    """

    if not isinstance(filename, basestring):
        raise TypeError('Filename must be a string')

    # try to infer the file format from the extension
    _, fmt = os.path.splitext(filename)

    if not fmt:
        fmt = '.json'
        filename = filename+fmt

    if fmt not in constants.EXPORT_FORMATS:
        raise ValueError('Export format not recognised')

    if fmt == '.json':
        export_JSON(filename, tracks)
    elif fmt == '.mat':
        export_MATLAB(filename, tracks)
    elif fmt == '.hdf5':
        export_HDF(filename, tracks)
    else:
        raise Exception('How did we get here?')


def check_track_type(tracks):
    return isinstance(tracks[0], core.Tracklet)





def export_JSON(filename, tracks):
    """ JSON Exporter for track data. """

    if not check_track_type(tracks):
        raise TypeError('Tracks must be of type core.Tracklet')

    # make a list of all track object data, sorted by track ID
    d = {"Tracklet_"+str(trk.ID):trk.to_dict() for trk in tracks}
    json_export = OrderedDict(sorted(d.items(), key=lambda t: t[1]['ID']))

    with open(filename, 'w') as json_file:
        json.dump(json_export, json_file, indent=2, separators=(',', ': '))



def export_MATLAB(filename, tracks):
    """ MATLAB Exporter for track data. """

    if not check_track_type(tracks):
        raise TypeError('Tracks must be of type core.Tracklet')

    from scipy.io import savemat
    np_cat = np.vstack([trk.to_array() for trk in tracks])
    matlab_export = {"num_tracks":len(tracks), "tracks": np_cat}
    savemat(filename, matlab_export)


def export_HDF(filename, tracks):
    """ HDF exporter for large datasets.

    This needs to deal with two different scenarios:
        i)  The original data came from an HDF5 file, in which case the file
            should exist and tracks should be a list of references, or
        ii) The original data came from another source, and we need to create
            the entire HDF5 file structure, including the objects data

    Args:
        filename - a string representing the HDF5 file
        tracks - either a list of refs or a list of core.Tracklet objects

    Notes:
        None

    """

    # check to see whether we have a list of references
    if isinstance(tracks[0], list):
        # ok, we have a list of object references
        if not os.path.exists(filename):
            raise IOError('HDF5 file does not exist: {0:s}'.format(filename))

        if not isinstance(tracks[0][0], long):
            print type(tracks[0][0]), tracks[0][0]
            raise TypeError('Track references should be integers')


        h = HDF5_FileHandler(filename)
        h.write_tracks(tracks)


    elif check_track_type(tracks):
        # we have a list of tracklet objects
        print 'oops!'

    else:
        raise TypeError('Tracks is of an unknown format.')










class HDF5_FileHandler(object):
    """ HDF5_FileHandler

    Generic HDF5 file hander for reading and writing datasets. This is
    inter-operable between segmentation, tracking and analysis code.

    Basic format of the HDF file is:
        frames/
            frame_1/
                ->object_1/
                ->object_2/
                ...
        segmented/
            segmented_1
            ...
        objects/
            object_1/
                x
                y
                z
                t
                ...
            object_2/
            ...
        tracks/
            track_1/
                ->object_12
                ->object_23
                ->object_74
                ...
            track_2/
            ...
        trees/

    Args:

    Members:

    Notes:

    """

    def __init__(self, filename=None):
        """ Initialise the HDF file. """
        if not filename.endswith('.hdf5'):
            filename+'.hdf5'
        self.filename = filename

        logging.info('Opening HDF file: {0:s}'.format(filename))
        self._hdf = h5py.File(filename, 'r+') # a -file doesn't have to exist

    def __del__(self):
        if self._hdf:
            self._hdf.close()
            logging.info('Closing HDF file.')

    @property
    def objects(self):
        """ Return the objects in the file """
        objects = [self.new_PyTrackObject(o) for o in self._hdf['objects']]
        return objects

    @property
    def tracks(self):
        """ Return the tracks in the file """
        tracks = [self.new_Tracklet(t) for t in self._hdf['tracks']]
        return tracks


    def write_tracks(self, tracks):
        """ Write the tracks to the HDF5 file """

        if 'tracks' not in self._hdf:
            self._hdf.create_group('tracks')
        else:
            logging.warning('Overwriting tracks in HDF5 file...')
            del self._hdf['tracks']
            self._hdf.create_group('tracks')

        trk_grp = self._hdf['tracks']

        # write out the tracks
        for t_ID, t in enumerate(tracks):
            trk_grp.create_dataset('track_{0:d}'.format(t_ID), data=np.array(t),
                dtype='i4')

    def write_objects(self, objects):
        """ Write objects to the HDF5 file """
        raise NotImplementedError


    def new_PyTrackObject(self, ID):
        """ Set up a new PyTrackObject quickly using data from a file """

        txyz = self._hdf['objects'][ID]['txyz']

        new_object = core.PyTrackObject()
        new_object.ID = ID_from_name(ID)
        new_object.t = txyz[0]
        new_object.x = txyz[1]
        new_object.y = txyz[2]
        new_object.z = txyz[3]
        new_object.dummy = False
        new_object.label = 0    # TODO(arl): grab these from the classifier
        new_object.probability = np.zeros((1,))
        return new_object

    def new_Tracklet(self, ID):
        """ Set up a new Tracklet quickly using data from a file """
        # TODO(arl): this needs cleaning up!!!
        n_obj = len(self._hdf['tracks'][ID])
        data = np.zeros((n_obj, 4))

        # traverse the softlinks and recreate the track
        for i, o in enumerate(self._hdf['tracks'][ID]):
            data[i,:] = self._hdf['tracks'][ID][o]

        new_track = core.Tracklet(ID_from_name(ID), data)
        return new_track






def import_HDF(filename):
    """ Import the HDF data.
    """

    hdf_handler = HDF5_FileHandler(filename)
    return hdf_handler.objects


def ID_from_name(name):
    """ Return the object ID from a name.

    'Object_203622' returns 203622
    """

    if not isinstance(name, basestring):
        raise TypError('object name must be of type string')

    m = re.search('(?<=_)\d+',name)
    # TODO(arl) some error checking here, incase the name is malformed
    return int(m.group(0))

def name_from_ID(ID):
    """ Return the name using an object ID.
    203622 returns 'Object_203622'
    """

    if not isinstance(ID, (int, long)):
        raise TypeError('object name must be of type integer')

    return 'object_{0:d}'.format(ID)



def import_JSON_observations(filename, labeller=None):
    """ import_JSON_observations

    Open a JSON file containing PyTrackObject objects

    Basic format is:
        {
          "Object_203622": {
            "x": 554.29737483861709,
            "y": 1199.362071438818,
            "z": 0.0,
            "t": 862,
            "label": "interphase",
            "states": 5,
            "probability": [
              0.996992826461792,
              0.0021888131741434336,
              0.0006106126820668578,
              0.000165432647918351,
              4.232166247675195e-05
            ],
            "dummy": false
          }
        }

    Args:
        filename: the filename of the JSON file
        labeller: an instance of the Labeller class

    Notes:
        None

    """

    # set a labeller
    if not labeller: labeller = Labeller()

    trk_objects = []

    with open(filename, 'r') as json_file:
        objects = json.load(json_file)

        itern = 0
        object_IDs = objects.keys()

        while objects:
            ID = object_IDs.pop()
            obj = objects.pop(ID)
            trk_obj = core.PyTrackObject()
            trk_obj.ID = ID_from_name(ID)
            for param in obj.keys():
                if param == 'probability':
                    # TODO(arl): clean this up. prob need to change JSON
                    # writer to prevent this outcome rather than fix here
                    if isinstance(obj['probability'], list):
                        trk_obj.probability = np.array(obj['probability'],dtype='float')
                    else:
                        trk_obj.probability = np.array((),dtype='float')
                    # append the number of states
                    trk_obj.states = len(obj['probability'])
                elif param == 'label':
                    trk_obj.label = labeller(obj['label'])
                else:
                    setattr(trk_obj, param, obj[param])

            trk_objects.append(trk_obj)

    return trk_objects




def import_ThunderSTORM(filename):
    """  Load localisation data from ThunderSTORM. """

    with open(filename, 'rb') as csvfile:
        localisations = csv.reader(csvfile, delimiter=',', quotechar='"')
        header = localisations.next()

        to_use = ('frame', 'x [nm]', 'y [nm]', 'z [nm]')
        cols = [i for i in xrange(len(header)) if header[i] in to_use]

        logging.info('Found: {0:s}'.format(', '.join([header[c] for c in cols])))

        objects = []

        #make a new object
        for data in localisations:
            obj = core.PyTrackObject()
            obj.t = int(float( data[cols[0]] ))
            obj.x = float(data[cols[1]])
            obj.y = float(data[cols[2]])
            if 'z [nm]' in header:
                obj.z = float(data[cols[3]])
            else:
                obj.z = 0.
            obj.dummy = False
            obj.label = 0
            obj.prob = np.zeros((5,),dtype='float')

            objects.append(obj)
    return objects






if __name__ == "__main__":
    import_HDF('/media/arl/Data/Work/test/MDCK/segmented.hdf5')
