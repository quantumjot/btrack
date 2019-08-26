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
__email__ = "code@arlowe.co.uk"

import re
import os
import numpy as np
import time
import csv
import json

import logging

# import core
import btypes
import constants

from collections import OrderedDict
from scipy.io import savemat



# get the logger instance
logger = logging.getLogger('worker_process')


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



def log_error(err_code):
    """ Take an error code from the tracker and log an error for the user.

    #define SUCCESS 900
    #define ERROR_empty_queue 901
    #define ERROR_no_tracks 902
    #define ERROR_no_useable_frames 903

    """

    if err_code in constants.ERRORS:
        logger.error('ERROR: {0:s}'.format(constants.ERRORS[err_code]))
        return True

    return False



def log_stats(stats):
    """ Take the statistics from the track and log the output """

    if log_error(stats['error']): return

    logger.info(' - Timing (Bayesian updates: {0:.2f}ms, Linking:'
                ' {1:.2f}ms)'.format(stats['t_update_belief'],
                stats['t_update_link']))

    logger.info(' - Probabilities (Link: {0:.5f}, Lost:'
                ' {1:.5f})'.format(stats['p_link'], stats['p_lost']))

    if stats['complete']:
        return

    logger.info(' - Stats (Active: {0:d}, Lost: {1:d}, Conflicts '
                'resolved: {2:d})'.format(stats['n_active'],
                stats['n_lost'], stats['n_conflicts']))



# def read_motion_model(filename):
def read_motion_model(config):
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
        model: a btypes.MotionModel instance for passing to BayesianTracker

    Notes:
        Note that the matrices are stored as 1D matrices here. In the future,
        this could form part of a Python only motion model.

        TODO(arl): More parsing of the data/reshaping arrays. Raise an
        appropriate error if there is something wrong with the model definition.
    """
    matrices = frozenset(['A','H','P','G','R'])
    model = btypes.MotionModel()

    # with open(filename, 'r') as j:
    #     config = json.load(j)

    if 'MotionModel' not in config.keys():
        raise ValueError('Not a valid motion model file')

    m = config['MotionModel']
    if not m: return None

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



# def read_object_model(filename):
def read_object_model(config):
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

    # with open(filename, 'r') as j:
    #     config = json.load(j)

    m = config['ObjectModel']
    if not m: return None

    matrices = frozenset(['transition','emission','start'])
    model = core.ObjectModel()

    if 'ObjectModel' not in config.keys():
        raise ValueError('Not a valid object model file')

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
    axes = ['x','y','z','t']
    within = lambda o: all([getattr(o, a)>=v[0] and getattr(o, a)<=v[1] for a,v in zip(axes, volume)])
    return [o for o in objects if within(o)]



def import_HDF(filename):
    """ Import the HDF data. """
    import dataio
    try:
        hdf_handler = dataio.HDF5_FileHandler(filename)
    except:
        hdf_handler = dataio.HDF5_FileHandler_LEGACY(filename)
    return hdf_handler.objects







if __name__ == "__main__":
    pass
