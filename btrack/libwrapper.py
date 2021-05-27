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
__email__ = "a.lowe@ucl.ac.uk"

import ctypes
import logging
import os
import platform

import numpy as np

from .btypes import PyTrackingInfo, PyTrackObject
from .constants import BTRACK_PATH
from .optimise import hypothesis

# get the logger instance
logger = logging.getLogger('worker_process')


def numpy_pointer_decorator(func):
    """ simple decorator for numpy ctypes pointers """
    return func()


@numpy_pointer_decorator
def np_dbl_p():
    """ Temporary function. Will remove in final release """
    return np.ctypeslib.ndpointer(
        dtype=np.double, ndim=2, flags='C_CONTIGUOUS'
    )


@numpy_pointer_decorator
def np_dbl_pc():
    """ Temporary function. Will remove in final release """
    return np.ctypeslib.ndpointer(
        dtype=np.double, ndim=2, flags='F_CONTIGUOUS'
    )


@numpy_pointer_decorator
def np_uint_p():
    """ Temporary function. Will remove in final release """
    return np.ctypeslib.ndpointer(
        dtype=np.uint32, ndim=2, flags='C_CONTIGUOUS'
    )


@numpy_pointer_decorator
def np_int_p():
    """ Temporary function. Will remove in final release """
    return np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')


@numpy_pointer_decorator
def np_int_vec_p():
    """ Temporary function. Will remove in final release """
    return np.ctypeslib.ndpointer(
        dtype=np.int32, ndim=1
    )  # , flags='C_CONTIGUOUS')


def load_library(filename):
    """Return the platform for shared library loading.

    Parameters
    ----------
    filename : str
        Filename for the shared library.

    Returns
    -------
    lib : ctypes.CDLL
        The ctypes `btrack` library.
    """

    if not isinstance(filename, str):
        raise TypeError('Filename must be a string')

    lib_file, ext = os.path.splitext(filename)

    system = platform.system()

    file_ext = {'Linux': '.so', 'Darwin': '.dylib', 'Windows': '.DLL'}
    full_lib_file = lib_file + file_ext[system]

    try:
        lib = ctypes.cdll.LoadLibrary(full_lib_file)
        logger.info(f'Loaded btrack: {full_lib_file}')
    except IOError:
        raise IOError(f'Cannot load shared library {full_lib_file}')

    return lib


def get_library():
    """Loads and returns the btrack shared library."""
    lib = load_library(os.path.join(BTRACK_PATH, 'libs', 'libtracker'))

    # deal with constructors/destructors
    lib.new_interface.restype = ctypes.c_void_p
    lib.new_interface.argtypes = [ctypes.c_bool]

    lib.del_interface.restype = None
    lib.del_interface.argtypes = [ctypes.c_void_p]

    # check the version number
    lib.check_library_version.restype = ctypes.c_bool
    lib.check_library_version.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
    ]

    # set the update method
    lib.set_update_mode.restype = None
    lib.set_update_mode.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    # set the motion model
    lib.motion.restype = None
    lib.motion.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        np_dbl_p,
        np_dbl_p,
        np_dbl_p,
        np_dbl_p,
        np_dbl_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint,
        ctypes.c_double,
    ]

    # set the object model
    # lib.model.restype = None
    # lib.model.argtypes = [ctypes.c_void_p, ctypes.c_uint, np_dbl_p,
    #                         np_dbl_p, np_dbl_p]

    lib.max_search_radius.restype = None
    lib.max_search_radius.argtypes = [ctypes.c_void_p, ctypes.c_float]

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
    lib.get.argtypes = [ctypes.c_void_p, np_dbl_p, ctypes.c_uint]

    # get the internal ID of a track
    lib.get_ID.restype = ctypes.c_uint
    lib.get_ID.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    # get a track, by reference
    lib.get_refs.restype = ctypes.c_uint
    lib.get_refs.argtypes = [ctypes.c_void_p, np_int_vec_p, ctypes.c_uint]

    # get the parent ID (i.e. pre-division)
    lib.get_parent.restype = ctypes.c_uint
    lib.get_parent.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    # get the root ID (i.e. the root ID of this lineage tree)
    lib.get_root.restype = ctypes.c_uint
    lib.get_root.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    # get the ID of any children
    lib.get_children.restype = ctypes.c_uint
    lib.get_children.argtypes = [ctypes.c_void_p, np_int_vec_p, ctypes.c_uint]

    # get the fate of the track
    lib.get_fate.restype = ctypes.c_uint
    lib.get_fate.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    # get the generational depth of the track
    lib.get_generation.restype = ctypes.c_uint
    lib.get_generation.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    # get the kalman filtered position
    lib.get_kalman_mu.restype = ctypes.c_uint
    lib.get_kalman_mu.argtypes = [ctypes.c_void_p, np_dbl_p, ctypes.c_uint]

    # get the kalman covariance
    lib.get_kalman_covar.restype = ctypes.c_uint
    lib.get_kalman_covar.argtypes = [ctypes.c_void_p, np_dbl_p, ctypes.c_uint]

    # get the predicted position at each time step
    lib.get_kalman_pred.restype = ctypes.c_uint
    lib.get_kalman_pred.argtypes = [ctypes.c_void_p, np_dbl_p, ctypes.c_uint]

    # get the label of the object
    lib.get_label.restype = ctypes.c_uint
    lib.get_label.argtypes = [ctypes.c_void_p, np_uint_p, ctypes.c_uint]

    # get the imaging volume
    lib.get_volume.restype = None
    lib.get_volume.argtypes = [ctypes.c_void_p, np_dbl_p]

    # get the imaging volume
    lib.set_volume.restype = None
    lib.set_volume.argtypes = [ctypes.c_void_p, np_dbl_p]

    # return a dummy object by reference
    lib.get_dummy.restype = PyTrackObject
    lib.get_dummy.argtypes = [ctypes.c_void_p, ctypes.c_int]

    # get the number of tracks
    lib.size.restype = ctypes.c_uint
    lib.size.argtypes = [ctypes.c_void_p]

    # calculate the hypotheses
    lib.create_hypotheses.restype = ctypes.c_uint
    lib.create_hypotheses.argtypes = [
        ctypes.c_void_p,
        hypothesis.PyHypothesisParams,
        ctypes.c_uint,
        ctypes.c_uint,
    ]

    # get a hypothesis by ID
    lib.get_hypothesis.restype = hypothesis.Hypothesis
    lib.get_hypothesis.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    # merge following optimisation
    lib.merge.restype = None
    lib.merge.argtypes = [ctypes.c_void_p, np_uint_p, ctypes.c_uint]

    return lib
