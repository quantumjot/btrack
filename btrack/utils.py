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
from . import btypes
from . import constants
from .optimise import hypothesis

from collections import OrderedDict
from scipy.io import savemat



# get the logger instance
logger = logging.getLogger('worker_process')


def load_config(filename):
    """ Load a tracking configuration file """
    if not os.path.exists(filename):
        # check whether it exists in the user model directory
        _, fn = os.path.split(filename)
        local_filename = os.path.join(constants.USER_MODEL_DIR, fn)

        if not os.path.exists(local_filename):
            logger.error("Configuration file {} not found".format(filename))
            raise IOError("Configuration file {} not found".format(filename))
        else:
            filename = local_filename

    with open(filename, 'r') as config_file:
        config = json.load(config_file)

    if "TrackerConfig" not in config:
        logger.error("Configuration file is malformed.")
        raise Exception("Tracking config is malformed")

    config = config["TrackerConfig"]

    logger.info("Loading configuration file: {}".format(filename))
    t_config = {"MotionModel": read_motion_model(config),
                "ObjectModel": read_object_model(config),
                "HypothesisModel": hypothesis.read_hypothesis_model(config)}

    return t_config



def log_error(err_code):
    """ Take an error code from the tracker and log an error for the user. """
    error = constants.Errors(err_code)
    if error != constants.Errors.SUCCESS and error != constants.Errors.NO_ERROR:
        logger.error('ERROR: {0:s}'.format(error))
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

    if 'MotionModel' not in list(config.keys()):
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

    m = config['ObjectModel']
    if not m: return None

    matrices = frozenset(['transition','emission','start'])
    model = core.ObjectModel()

    if 'ObjectModel' not in list(config.keys()):
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


def import_HDF(filename, filter_using=None):
    """ Import the HDF data. """
    from . import dataio
    with dataio.HDF5FileHandler(filename) as hdf:
        objects = hdf.filtered_objects(f_expr=filter_using)
    return objects

def import_JSON(filename):
    """ Import JSON data """
    from . import dataio
    return dataio.import_JSON(filename)


def build_trees(tracks, update_tracks=True):
    """ Build lineage trees and update the track relationships """
    from btrack.optimise import lineage
    tree = lineage.LineageTree(tracks)
    return tree.create(update_tracks=update_tracks)



if __name__ == "__main__":
    pass
