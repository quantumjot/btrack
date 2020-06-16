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
import csv
import h5py
import json
import logging

# import core
from . import btypes
from . import constants

import numpy as np

from collections import OrderedDict
from functools import wraps



# get the logger instance
logger = logging.getLogger('worker_process')


class _PyTrackObjectFactory:
    def __init__(self):
        self.reset()

    def get(self, txyz, label=None, obj_type=0):
        """ get an instatiated object """
        assert(isinstance(txyz, np.ndarray))
        assert(txyz[0] >= 0.) # assert that time is always positive!
        if label is not None:
            if isinstance(label, int):
                class_label = label
                probability = np.zeros((1,))
            else:
                class_label = label[0].astype(np.uint32)
                probability = label[1:].astype(np.float32)
        else:
            class_label = constants.States.NULL.value
            probability = np.zeros((1,))

        new_object = btypes.PyTrackObject()
        new_object.ID = self._ID
        new_object.t = txyz[0].astype(np.uint32)
        new_object.x = txyz[1].astype(np.float32)
        new_object.y = txyz[2].astype(np.float32)
        new_object.z = txyz[3].astype(np.float32)
        new_object.dummy = False
        new_object.label = class_label          # from the classifier
        new_object.probability = probability
        new_object.type = int(obj_type)

        self._ID += 1
        return new_object

    def reset(self):
        self._ID = 0

# instatiate the factory
ObjectFactory = _PyTrackObjectFactory()


def import_JSON(filename):
    """ generic JSON importer for localisations from other software """
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    objects = []
    for ID, _obj in data.items():
        txyz = np.array([_obj[k] for k in ['t','x','y','z']])
        obj = ObjectFactory.get(txyz, label=int(_obj['label']))
        objects.append(obj)
    return objects


def import_CSV(filename):
    """ import from a CSV file

    CSV file should have one of the following formats:

    t, x, y
    t, x, y, label
    t, x, y, z
    t, x, y, z, label

    """
    objects = []
    with open(filename, 'r') as csv_file:
        csvreader = csv.DictReader(csvfile, delimiter=' ', quotechar='|')
        for row in csvreader:
            txyz = np.zeros((1,4), dtype=np.float32)
            txyz[:,(0, 1, 2)] = [row[k] for k in ('t', 'x', 'y')]
            if 'z' in row:
                txyz[:,3] = row['z'] # if we have z info
            if 'label' in row:
                label = int(row['label'])
            else:
                label = None

            objects.append(ObjectFactory.get(txyz, label=label))
    return objects




def export_delegator(filename, tracker, obj_type=None, filter_by=None):
    """ Export data from the tracker using the appropriate exporter """
    # assert(isinstance(tracker, BayesianTracker))
    assert(isinstance(filename, str))

    export_dir, export_fn = os.path.split(filename)
    _, ext = os.path.splitext(filename)

    assert(os.path.exists(export_dir))

    if ext == '.json':
        raise DeprecationWarning('JSON export is deprecated')
    elif ext == '.mat':
        raise DeprecationWarning('MATLAB export is deprecated')
    elif ext == '.csv':
        export_CSV(filename, tracker.tracks, obj_type=obj_type)
    elif ext in ('.hdf', '.hdf5', '.h5'):
        _export_HDF(filename, tracker, obj_type=obj_type, filter_by=filter_by)
    else:
        logger.error(f'Export file format {ext} not recognized.')





def check_track_type(tracks):
    # return isinstance(tracks[0], btypes.Tracklet)
    return all([isinstance(t, btypes.Tracklet) for t in tracks])



def export_CSV(filename: str,
               tracks: list,
               properties: list = constants.DEFAULT_EXPORT_PROPERTIES,
               obj_type=None):
    """ export the track data as a simple CSV file """

    if not tracks:
        logger.error(f'No tracks found when exporting to: {filename}')
        return

    if not check_track_type(tracks):
        logger.error('Tracks of incorrect type')

    logger.info(f'Writing out CSV files to: {filename}')
    export_track = np.vstack([t.to_array(properties) for t in tracks])

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        csvwriter.writerow(properties)
        for i in range(export_track.shape[0]):
            csvwriter.writerow(export_track[i,:].tolist())




def export_LBEP(filename: str, tracks: list):
    """ export the LBEP table described here:
    https://public.celltrackingchallenge.net/documents/
        Naming%20and%20file%20content%20conventions.pdf
    """
    if not tracks:
        logger.error(f'No tracks found when exporting to: {filename}')
        return

    if not check_track_type(tracks):
        logger.error('Tracks of incorrect type')

    tracks.sort(key=lambda t: t.ID)
    if not filename.endswith('.txt'): filename+='.txt'
    with open(filename, 'w') as lbep_file:
        logger.info(f'Writing LBEP file: {filename}...')
        for track in tracks:
            lbep = f'{track.ID} {track.t[0]} {track.t[-1]} {track.parent}'
            lbep_file.write(f'{lbep}\n')


def _export_HDF(filename: str,
               tracker,
               obj_type=None,
               filter_by: str = None):
    """ export to HDF """
    with HDF5FileHandler(filename, read_write='a') as hdf:
        hdf.write_tracks(tracker, obj_type=obj_type, f_expr=filter_by)



def h5check_property_exists(property):
    """ Wrapper for hdf handler to make sure a property exists """
    def func(fn):
        @wraps(fn)
        def wrapped_handler_property(*args, **kwargs):
            self = args[0]
            assert(isinstance(self, HDF5FileHandler))
            if property not in self._hdf:
                logger.error(f'{property.capitalize()} not found in {self.filename}')
                return None
            return fn(*args, **kwargs)
        return wrapped_handler_property
    return func



class HDF5FileHandler:
    """ HDF5FileHandler

    Generic HDF5 file hander for reading and writing datasets. This is
    inter-operable between segmentation, tracking and analysis code.

    LBEPR is a modification of the LBEP format to also include the root node
    of the tree.

        I - number of objects
        J - number of frames
        K - number of tracks

    Added generic filtering to object retrieval, e.g.
        obj = handler.filtered_objects('flag==1')
        retrieves all objects if there is an object['flag'] == 1

    Basic format of the HDF file is:
        segmentation/
            images          - (J x h x w) uint8 images of the segmentation
        objects/
            obj_type_1/
                coords      - (I x 5) [t, x, y, z, object_type]
                labels      - (I x D) [label, (softmax scores ...)]
                map         - (J x 2) [start_index, end_index] -> coords array
            ...
        tracks/
            obj_type_1/
                tracks      - (I x 1) [index into coords]
                dummies     - similar to coords, but for dummy objects
                map         - (K x 2) [start_index, end_index] -> tracks array
                LBEPRG      - (K x 6) [L, B, E, P, R, G]
                fates       - (K x n) [fate_from_tracker, ...future_expansion]
            ...

    Notes:
        NOTE(arl): the final slice [:] reads the whole file in one go,
        since we are unlikely to have more objects than memory and we
        need to load them all anyway.

        NOTE(arl): should dummies be moved to coords?
    """

    def __init__(self, filename, read_write='r'):
        self.filename = filename
        logger.info(f'Opening HDF file: {self.filename}')
        self._hdf = h5py.File(filename, read_write)
        self._states = list(constants.States)
        self._f_expr = None # DO NOT USE

    @property
    def object_types(self):
        return list(self._hdf['objects'].keys())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self._hdf: return
        logger.info(f'Closing HDF file: {self.filename}')
        self._hdf.close()

    @property
    def objects(self):
        """ Return the objects in the file """
        return self.filtered_objects()

    @h5check_property_exists('objects')
    def filtered_objects(self, f_expr=None, obj_types=None):
        """ return a filtered list of objects based on metadata.
        f_expr should be of the format 'flag==1'
        """

        objects = []
        if obj_types is None:
            obj_types = self.object_types
        else:
            assert(isinstance(obj_types, list))
            assert(all([o in self.object_types for o in obj_types]))

        for ci, c in enumerate(obj_types):
            # read the whole dataset into memory
            txyz = self._hdf['objects'][c]['coords'][:]
            if 'labels' not in self._hdf['objects'][c]:
                logger.warning('Labels missing from objects in HDF file')
                labels = np.zeros((txyz.shape[0], 6))
            else:
                labels = self._hdf['objects'][c]['labels'][:]
            idx = range(txyz.shape[0])      # default filtering uses all objects

            # note that this doesn't do much error checking at the moment
            if f_expr is not None:
                assert(isinstance(f_expr, str))
                pattern = '(?P<name>\w+)(?P<op>[\>\<\=]+)(?P<cmp>[0-9]+)'
                m = re.match(pattern, f_expr)
                f_eval = f'x{m["op"]}{m["cmp"]}' # e.g. x > 10

                if m['name'] in self._hdf['objects'][c]:
                    data = self._hdf['objects'][c][m['name']][:]
                    idx = [i for i, x in enumerate(data) if eval(f_eval)]
                else:
                    logger.warning(f'Cannot filter objects by {m["name"]}')

            assert(txyz.shape[0] == labels.shape[0])
            logger.info(f'Loading {c} {txyz.shape} ({len(idx)} filtered: {f_expr})...')
            obj = [ObjectFactory.get(txyz[i,:], label=labels[i,:], obj_type=ci+1) for i in idx]
            objects += obj
        return objects

    def write_tracks(self, tracker, obj_type=None, f_expr=None):
        """ Write tracks to HDF file """
        if not tracker.tracks:
            logger.error(f'No tracks found when exporting to: {self.filename}')
            return

        assert(obj_type in self.object_types)

        tracks = tracker.refs
        dummies = tracker.dummies
        lbep_table = np.stack(tracker.lbep, axis=0).astype(np.int32)

        # sanity check
        assert(lbep_table.shape[0] == len(tracks))

        logger.info(f'Writing tracks to HDF file: {self.filename}')
        hdf_tracks = np.concatenate(tracks, axis=0)

        hdf_frame_map = np.zeros((len(tracks),2), dtype=np.int32)
        for i, track in enumerate(tracks):
            if i > 0:
                offset = hdf_frame_map[i-1,1]
            else: offset = 0
            hdf_frame_map[i,:] = np.array([0, len(track)]) + offset

        if 'tracks' not in self._hdf:
            self._hdf.create_group('tracks')

        if obj_type in self._hdf['tracks']:
            logger.warning(f'Removing {obj_type} from HDF file. ')
            del self._hdf['tracks'][obj_type]

        grp = self._hdf['tracks'].create_group(obj_type)
        grp.create_dataset('tracks', data=hdf_tracks, dtype='int32')
        grp.create_dataset('map', data=hdf_frame_map, dtype='uint32')

        # if we have used the f_expr we can save it as an attribute here
        if f_expr is not None and isinstance(f_expr, str):
            grp.attrs['f_expr'] = f_expr

        # also save the version number as an attribute
        grp.attrs['version'] = constants.get_version()

        # write out dummies
        if dummies:
            logger.info(f'Writing dummies to HDF file: {self.filename}')
            o = self.object_types.index(obj_type) + 1
            txyz = np.stack([[d.t, d.x, d.y, d.z, o] for d in dummies], axis=0)
            grp.create_dataset('dummies', data=txyz, dtype='float32')

        # write out the LBEP table
        logger.info(f'Writing LBEPR to HDF file: {self.filename}')
        grp.create_dataset('LBEPR', data=lbep_table, dtype='int32')

        # write out cell fates
        logger.info(f'Writing track fates to HDF file: {self.filename}')
        fate_table = np.stack([t.fate.value for t in tracker.tracks], axis=0)
        grp.create_dataset('fates', data=fate_table, dtype='int32')

    @property
    @h5check_property_exists('tracks')
    def tracks(self):
        """ Return the tracks in the file """
        dummies, ret = [], []

        # make an object factory
        factory = _PyTrackObjectFactory()

        for ci, c in enumerate(self._hdf['tracks'].keys()):
            logger.info(f'Loading tracks: {c}...')
            track_map = self._hdf['tracks'][c]['map'][:]
            track_refs = self._hdf['tracks'][c]['tracks'][:]
            lbep = self._hdf['tracks'][c]['LBEPR'][:]
            fates = self._hdf['tracks'][c]['fates'][:]

            # if there are dummies, make new dummy objects
            if 'dummies' in self._hdf['tracks'][c]:
                dummies = self._hdf['tracks'][c]['dummies'][:]
                n_dummies = dummies.shape[0]
                dobj = [factory.get(dummies[i,:]) for i in range(n_dummies)]
                for d in dobj: d.dummy = True

            # TODO(arl): this needs to be stored in the HDF folder
            if 'f_expr' in self._hdf['tracks'][c].attrs:
                f_expr = self._hdf['tracks'][c].attrs['f_expr']
            elif self._f_expr is not None:
                f_expr = self._f_expr
            else:
                f_expr = None

            obj = self.filtered_objects(f_expr=f_expr, obj_types=[c])

            def get_txyz(ref):
                if ref>=0: return obj[ref]
                return dobj[abs(ref)-1] # references are -ve for dummies

            tracks = []
            for i in range(track_map.shape[0]):
                idx = slice(*track_map[i,:].tolist())
                refs = track_refs[idx]
                track = btypes.Tracklet(lbep[i,0], list(map(get_txyz, refs)))
                track.parent = lbep[i,3]    # set the parent and root of tree
                track.root = lbep[i,4]
                if lbep.shape[1] > 5: track.generation = lbep[i,5]
                track.fate = constants.Fates(fates[i]) # restore the track fate
                tracks.append(track)

            # once we have all of the tracks, populate the children
            to_update = {}
            for track in tracks:
                if not track.is_root:
                    parents = filter(lambda t: t.ID == track.parent, tracks)
                    for parent in parents:
                        if parent not in to_update:
                            to_update[parent] = []
                        to_update[parent].append(track.ID)

            # sanity check, can be removed at a later date
            assert all([len(children)<=2 for children in to_update.values()])

            # add the children to the parent
            for track, children in to_update.items():
                track.children = children

            ret.append(tracks)

        return ret

    @property
    def segmentation(self):
        raise NotImplementedError

    @property
    @h5check_property_exists('tracks')
    def lbep(self):
        logger.info('Loading LBEPR tables...')
        return [self._hdf['tracks'][k]['LBEPR'][:] for k in self._hdf['tracks']]






if __name__ == "__main__":
    pass
