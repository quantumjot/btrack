from __future__ import annotations

import itertools
import logging
import os
import re
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import h5py
import numpy as np

# import core
from btrack import btypes, constants, utils

from .utils import (
    check_object_type,
    check_track_type,
    objects_from_array,
    objects_from_dict,
)

if TYPE_CHECKING:
    from btrack import BayesianTracker

# get the logger instance
logger = logging.getLogger(__name__)


def h5check_property_exists(property):  # noqa: A002
    """Wrapper for hdf handler to make sure a property exists."""

    def func(fn):
        @wraps(fn)
        def wrapped_handler_property(*args, **kwargs):
            self = args[0]
            assert isinstance(self, HDF5FileHandler)
            if property not in self._hdf:
                logger.error(
                    f"{property.capitalize()} not found in {self.filename}"
                )
                return None
            return fn(*args, **kwargs)

        return wrapped_handler_property

    return func


class HDF5FileHandler:
    """Generic HDF5 file hander for reading and writing datasets. This is
    inter-operable between segmentation, tracking and analysis code.

    Parameters
    ----------
    filename : str
        The filename of the hdf5 file to be used.
    read_write : str
        A read/write mode for the file, e.g. `w`, `r`, `a` etc.
    obj_type : str
        The name of the object type. Defaults to `obj_type_1`. The object type
        name must start with `obj_type_`

    Attributes
    ----------
    segmentation : np.ndarray
        A numpy array representing the segmentation data. TZYX
    objects : list [PyTrackObject]
        A list of PyTrackObjects localised from the segmentation data.
    filtered_objects  : np.ndarray
        Similar to objects, but filtered by property.
    tracks : list [Tracklet]
        A list of Tracklet objects.
    lbep : np.ndarray
        The LBEP table representing the track graph.

    Notes
    -----
    ```
    Basic format of the HDF file is:
        segmentation/
            images          - (J x (d) x h x w) uint16 segmentation
        objects/
            obj_type_1/
                coords      - (I x 5) [t, x, y, z, object_type]
                labels      - (I x D) [label, (softmax scores ...)]
                map         - (J x 2) [start_index, end_index] -> coords array
                properties/
                    area  - (I x 1) first named property (e.g. `area`)
                    ...
            ...
        tracks/
            obj_type_1/
                tracks      - (I x 1) [index into coords]
                dummies     - similar to coords, but for dummy objects
                map         - (K x 2) [start_index, end_index] -> tracks array
                LBEPRG      - (K x 6) [L, B, E, P, R, G]
                fates       - (K x n) [fate_from_tracker, ...future_expansion]
            ...

    Where:
        I - number of objects
        J - number of frames
        K - number of tracks
    ```

    LBEPR is a modification of the LBEP format to also include the root node
    of the tree.


    Examples
    --------
    Read objects from a file:
    >>> with HDF5FileHandler('file.h5', 'r') as handler:
    >>>    objects = handler.objects

    Use filtering by property for object retrieval:
    >>> obj = handler.filtered_objects('flag==1')
    >>> obj = handler.filtered_objects('area>100')

    Write tracks directly to a file:
    >>> handler.write_tracks(tracks)
    """

    def __init__(
        self,
        filename: os.PathLike,
        read_write: str = "r",
        *,
        obj_type: str = "obj_type_1",
    ):
        self._f_expr = None  # DO NOT USE
        self.object_type = obj_type

        self.filename = filename
        logger.info(f"Opening HDF file: {self.filename}...")
        self._hdf = h5py.File(filename, read_write)
        self._states = list(constants.States)

    @property
    def object_types(self) -> List[str]:
        return list(self._hdf["objects"].keys())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self._hdf:
            return
        logger.info(f"Closing HDF file: {self.filename}")
        self._hdf.close()

    @property
    def object_type(self) -> str:
        return self._object_type

    @object_type.setter
    def object_type(self, obj_type: str) -> None:
        if not obj_type.startswith("obj_type_"):
            raise ValueError("Object type must start with ``obj_type_``")
        self._object_type = obj_type

    @property  # type: ignore
    @h5check_property_exists("segmentation")
    def segmentation(self) -> np.ndarray:
        segmentation = self._hdf["segmentation"]["images"][:].astype(np.uint16)
        logger.info(f"Loading segmentation {segmentation.shape}")
        return segmentation

    def write_segmentation(self, segmentation: np.ndarray) -> None:
        """Write out the segmentation to an HDF file.

        Parameters
        ----------
        segmentation : np.ndarray
            A numpy array representing the segmentation data. T(Z)YX, uint16
        """
        # write the segmentation out
        grp = self._hdf.create_group("segmentation")
        grp.create_dataset(
            "images",
            data=segmentation,
            dtype="uint16",
            compression="gzip",
            compression_opts=7,
        )

    @property
    def objects(self) -> List[btypes.PyTrackObject]:
        """Return the objects in the file."""
        return self.filtered_objects()

    @h5check_property_exists("objects")
    def filtered_objects(
        self,
        f_expr: Optional[str] = None,
        *,
        lazy_load_properties: bool = True,
        exclude_properties: Optional[List[str]] = None,
    ) -> List[btypes.PyTrackObject]:
        """A filtered list of objects based on metadata.

        Parameters
        ----------
        f_expr : str
            A string representing a filtering option. For example, `area>100`
            would filter objects by a property key `area` where the numerical
            value of area was greater than 100.
        lazy_load_properties : bool
            For future expansion. To allow lazy loading of large datasets.
        exclude_properties : list or None
            A list of properties keys to exclude when loading from disk.

        Returns
        -------
        objects : list
            A list of :py:class:`btrack.btypes.PyTrackObject` objects.
        """

        exclude_properties = (
            exclude_properties if exclude_properties is not None else []
        )

        if self.object_type not in self.object_types:
            raise ValueError(f"Object type {self.object_type} not recognized")

        grp = self._hdf["objects"][self.object_type]

        # read the whole dataset into memory
        txyz = grp["coords"][:]
        if "labels" not in grp:
            logger.warning("Labels missing from objects in HDF file")
            labels = np.zeros((txyz.shape[0], 6))
        else:
            labels = self._hdf["objects"][self.object_type]["labels"][:]

        # get properties if we have them (note, this assumes that the same
        # properties exist for each object)
        properties = {}
        if "properties" in grp:
            p_keys = list(
                set(grp["properties"].keys()).difference(
                    set(exclude_properties)
                )
            )
            properties = {k: grp["properties"][k][:] for k in p_keys}
            assert all([len(p) == len(txyz) for p in properties.values()])

        # note that this doesn't do much error checking at the moment
        # TODO(arl): this should now reference the `properties`
        if f_expr is not None:
            assert isinstance(f_expr, str)
            pattern = r"(?P<name>\w+)(?P<op>[\>\<\=]+)(?P<cmp>[0-9]+)"
            m = re.match(pattern, f_expr)

            if m is None:
                raise ValueError(f"Cannot filter objects by {f_expr}")

            f_eval = f"x{m['op']}{m['cmp']}"  # e.g. x > 10

            if m["name"] in properties:
                data = properties[m["name"]]
                filtered_idx = [i for i, x in enumerate(data) if eval(f_eval)]
            else:
                raise ValueError(f"Cannot filter objects by {f_expr}")

        else:
            filtered_idx = range(txyz.shape[0])  # default filtering uses all

        # sanity check that coordinates matches labels
        assert txyz.shape[0] == labels.shape[0]
        logger.info(
            f"Loading objects/{self.object_type} {txyz.shape} "
            f"({len(filtered_idx)} filtered: {f_expr})"
        )

        txyz_filtered = txyz[filtered_idx, :]
        labels_filtered = labels[filtered_idx, :]

        objects_dict = {
            "t": txyz_filtered[:, 0],
            "x": txyz_filtered[:, 1],
            "y": txyz_filtered[:, 2],
            "z": txyz_filtered[:, 3],
            "label": labels_filtered[:, 0],
            "ID": np.asarray(filtered_idx),
        }

        # add the filtered properties
        for key, props in properties.items():
            objects_dict.update({key: props[filtered_idx]})

        return objects_from_dict(objects_dict)

    def write_objects(
        self, data: Union[List[btypes.PyTrackObject], BayesianTracker]
    ) -> None:
        """Write objects to HDF file.

        Parameters
        ----------
        data : list or BayesianTracker instance
            Either a list of PyTrackObject to be written, or an instance of
            BayesianTracker with a .objects property.
        """
        # TODO(arl): make sure that the objects are ordered in time

        if isinstance(data, list):
            objects = data
        elif hasattr(data, "objects"):
            objects = data.objects
        else:
            raise TypeError("Object type not recognized.")

        # make sure that the data to be written are all of type PyTrackObject
        if not check_object_type(objects):
            raise TypeError("Object type not recognized.")

        if "objects" not in self._hdf:
            self._hdf.create_group("objects")
        grp = self._hdf["objects"].create_group(self.object_type)
        props = {k: [] for k in objects[0].properties}

        n_objects = len(objects)
        n_frames = np.max([o.t for o in objects]) + 1

        txyz = np.zeros((n_objects, 5), dtype=np.float32)
        labels = np.zeros((n_objects, 1), dtype=np.uint8)
        fmap = np.zeros((n_frames, 2), dtype=np.uint32)

        # convert the btrack objects into a numpy array
        for i, obj in enumerate(objects):
            txyz[i, :] = [obj.t, obj.x, obj.y, obj.z, 0]
            labels[i, :] = obj.label
            t = int(obj.t)
            fmap[t, 1] = np.max([fmap[t, 1], i])

            # add in any properties
            for key in props:
                props[key].append(obj.properties[key])

        fmap[1:, 0] = fmap[:-1, 1]

        logger.info(f"Writing objects/{self.object_type}")
        grp.create_dataset("coords", data=txyz, dtype="float32")
        grp.create_dataset("map", data=fmap, dtype="uint32")

        logger.info(f"Writing labels/{self.object_type}")
        grp.create_dataset("labels", data=labels, dtype="float32")

        # finally, write any properties
        self.write_properties(props)

    @h5check_property_exists("objects")
    def write_properties(
        self, data: Dict[str, Any], *, allow_overwrite: bool = False
    ) -> None:
        """Write object properties to HDF file.

        Parameters
        ----------
        data : dict {key: (N, D)}
            A dictionary of key-value pairs of properties to be written. The
            values should be an array equal in length to the number of objects
            and with D dimensions.
        allow_overwrite : bool
            Allow to delete the existing property keys from the HDF5 file and
            overwrite with new values from the data dict. Defaults to False.
        """

        if not isinstance(data, dict):
            raise TypeError("Properties must be a dictionary.")

        grp = self._hdf[f"objects/{self.object_type}"]

        if "properties" not in grp.keys():
            props_grp = grp.create_group("properties")
        else:
            props_grp = self._hdf[f"objects/{self.object_type}/properties"]

        n_objects = len(self.objects)

        for key, values in data.items():
            # Manage the property data:
            if not values:
                logger.warning(f"Property '{key}' is empty.")
                continue
            values = np.array(values)  # noqa: PLW2901
            assert values.shape[0] == n_objects

            # Check if the property is already in the props_grp:
            if key in props_grp:
                if allow_overwrite is False:
                    logger.info(
                        f"Property '{key}' already written in the file"
                    )
                    raise KeyError(
                        f"Property '{key}' already in file -> switch on "
                        "'overwrite' param to replace existing property "
                    )
                else:
                    del self._hdf[f"objects/{self.object_type}/properties"][
                        key
                    ]
                    logger.info(
                        f"Property '{key}' erased to be overwritten..."
                    )

            # Now that you handled overwriting, write the values:
            logger.info(
                f"Writing properties/{self.object_type}/{key} {values.shape}"
            )
            props_grp.create_dataset(key, data=data[key], dtype="float32")

    @property  # type: ignore
    @h5check_property_exists("tracks")
    def tracks(self) -> List[btypes.Tracklet]:
        """Return the tracks in the file."""

        logger.info(f"Loading tracks/{self.object_type}")
        track_map = self._hdf["tracks"][self.object_type]["map"][:]
        track_refs = self._hdf["tracks"][self.object_type]["tracks"][:]
        lbep = self.lbep
        fates = self._hdf["tracks"][self.object_type]["fates"][:]

        # if there are dummies, make new dummy objects
        if "dummies" in self._hdf["tracks"][self.object_type]:
            dummies = self._hdf["tracks"][self.object_type]["dummies"][:]
            dummy_obj = objects_from_array(dummies[:, :4])
            for d in dummy_obj:
                d.ID = -(d.ID + 1)  # restore the -ve ID
                d.dummy = True  # set the dummy flag to true

        # TODO(arl): this needs to be stored in the HDF folder
        if (
            "f_expr" in self._hdf["tracks"][self.object_type].attrs
            and self._f_expr is None
        ):
            f_expr = self._hdf["tracks"][self.object_type].attrs["f_expr"]
        elif self._f_expr is not None:
            f_expr = self._f_expr
        else:
            f_expr = None

        obj = self.filtered_objects(f_expr=f_expr)

        def _get_txyz(_ref: int) -> int:
            if _ref >= 0:
                return obj[_ref]
            return dummy_obj[abs(_ref) - 1]  # references are -ve for dummies

        tracks = []
        for i in range(track_map.shape[0]):
            idx = slice(*track_map[i, :].tolist())
            refs = track_refs[idx]
            track = btypes.Tracklet(lbep[i, 0], list(map(_get_txyz, refs)))
            track.parent = lbep[i, 3]  # set the parent and root of tree
            track.root = lbep[i, 4]
            if lbep.shape[1] > 5:  # noqa: PLR2004
                track.generation = lbep[i, 5]
            track.fate = constants.Fates(fates[i])  # restore the track fate
            tracks.append(track)

        # once we have all of the tracks, populate the children
        to_update = {}
        for track in tracks:
            if not track.is_root:
                parents = filter(lambda t: track.parent == t.ID, tracks)
                for parent in parents:
                    if parent not in to_update:
                        to_update[parent] = []
                    to_update[parent].append(track.ID)

        # sanity check, can be removed at a later date
        MAX_N_CHILDREN = 2
        assert all(
            len(children) <= MAX_N_CHILDREN for children in to_update.values()
        )

        # add the children to the parent
        for track, children in to_update.items():
            track.children = children

        return tracks

    def write_tracks(  # noqa: PLR0912
        self,
        data: Union[List[btypes.Tracklet], BayesianTracker],
        *,
        f_expr: Optional[str] = None,
    ) -> None:
        """Write tracks to HDF file.

        Parameters
        ----------
        data : list of Tracklets or an instance of BayesianTracker
            A list of tracklets or an instance of BayesianTracker.
        f_expr : str
            An expression which represents how the objects have been filtered
            prior to tracking, e.g. `area>100.0`
        """

        if isinstance(data, list):
            if not check_track_type(data):
                raise ValueError(f"Data of type {type(data)} not supported.")

            all_objects = itertools.chain.from_iterable(
                [trk._data for trk in data]
            )

            objects = [obj for obj in all_objects if not obj.dummy]
            dummies = [obj for obj in all_objects if obj.dummy]

            # renumber the object ID so that they can be stored in a contiguous
            # array and indexed by row - this may not be necessary for most
            # datasets, but is here just in case
            for idx, obj in enumerate(objects):
                obj.ID = idx

            for idx, dummy in enumerate(dummies):
                dummy.ID = -(idx + 1)

            refs = [trk.refs for trk in data]
            lbep_table = utils._lbep_table(data)
            fate_table = np.stack([t.fate.value for t in data], axis=0)

            if "objects" not in self._hdf:
                self.write_objects(objects)

        elif hasattr(data, "tracks"):
            refs = data.refs
            dummies = data.dummies
            lbep_table = data.LBEP
            fate_table = np.stack([t.fate.value for t in data.tracks], axis=0)
        else:
            raise ValueError(f"Data of type {type(data)} not supported.")

        if not refs:
            logger.error(f"No tracks found when exporting to: {self.filename}")
            return

        # sanity check
        assert lbep_table.shape[0] == len(refs)

        logger.info(f"Writing tracks/{self.object_type}")
        hdf_tracks = np.concatenate(refs, axis=0)

        hdf_frame_map = np.zeros((len(refs), 2), dtype=np.int32)
        for i, track in enumerate(refs):
            offset = hdf_frame_map[i - 1, 1] if i > 0 else 0
            hdf_frame_map[i, :] = np.array([0, len(track)]) + offset

        if "tracks" not in self._hdf:
            self._hdf.create_group("tracks")

        if self.object_type in self._hdf["tracks"]:
            logger.warning(f"Removing tracks/{self.object_type}.")
            del self._hdf["tracks"][self.object_type]

        grp = self._hdf["tracks"].create_group(self.object_type)
        grp.create_dataset("tracks", data=hdf_tracks, dtype="int32")
        grp.create_dataset("map", data=hdf_frame_map, dtype="uint32")

        # if we have used the f_expr we can save it as an attribute here
        if f_expr is not None and isinstance(f_expr, str):
            grp.attrs["f_expr"] = f_expr

        # also save the version number as an attribute
        grp.attrs["version"] = constants.get_version()

        # write out dummies
        if dummies:
            logger.info(f"Writing dummies/{self.object_type}")
            o = self.object_types.index(self.object_type) + 1
            txyz = np.stack([[d.t, d.x, d.y, d.z, o] for d in dummies], axis=0)
            grp.create_dataset("dummies", data=txyz, dtype="float32")

        # write out the LBEP table
        logger.info(f"Writing LBEP/{self.object_type}")
        grp.create_dataset("LBEPR", data=lbep_table, dtype="int32")

        # write out cell fates
        logger.info(f"Writing fates/{self.object_type}")
        grp.create_dataset("fates", data=fate_table, dtype="int32")

    @property  # type: ignore
    @h5check_property_exists("tracks")
    def lbep(self) -> np.ndarray:
        """Return the LBEP data."""
        logger.info(f"Loading LBEP/{self.object_type}")
        return self._hdf["tracks"][self.object_type]["LBEPR"][:]
