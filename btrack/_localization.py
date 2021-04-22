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
__email__ = "code@arlowe.co.uk"

import inspect
import logging
from typing import Generator, Optional, Tuple, Union

import numpy as np
from skimage.measure import label, regionprops_table

from .dataio import localizations_to_objects

# get the logger instance
logger = logging.getLogger("worker_process")


def _centroids_from_single_arr(
    segmentation: Union[np.ndarray, Generator],
    properties: Tuple[str],
    frame: int,
    intensity_image: Optional[np.ndarray] = None,
    scale: Optional[Tuple[float]] = None,
    use_weighted_centroid: bool = False,
) -> np.ndarray:
    """Return the object centroids from a numpy array representing the
    image data."""

    if np.sum(segmentation) == 0:
        return {}

    def _is_binary(x: np.ndarray) -> bool:
        return ((x == 0) | (x == 1)).all()

    if use_weighted_centroid and intensity_image is not None:
        CENTROID_PROPERTY = "weighted_centroid"
    else:
        CENTROID_PROPERTY = "centroid"

    if CENTROID_PROPERTY not in properties:
        properties = (CENTROID_PROPERTY,) + properties

    # check to see whether this is a binary segmentation
    # TODO(arl): of course, this may also be ternary etc, so this will
    # fail, should really check that the labels are unique
    if _is_binary(segmentation):
        labeled = label(segmentation)
    else:
        labeled = segmentation

    _centroids = regionprops_table(
        labeled, intensity_image=intensity_image, properties=properties,
    )

    # add time to the array
    _centroids['t'] = np.full(
        _centroids[f"{CENTROID_PROPERTY}-0"].shape, frame
    )

    # apply the anistropic scaling
    if scale is not None:
        if len(scale) != segmentation.ndim:
            raise ValueError("Scale dimensions do not match segmentation.")

        # perform the anistropic scaling
        for dim in range(segmentation.ndim):
            _centroids[f"{CENTROID_PROPERTY}-{dim}"] = np.multiply(
                _centroids[f"{CENTROID_PROPERTY}-{dim}"], float(scale[dim])
            )

    # now rename the axes for btrack
    dim_names = ["z", "y", "x"][-(segmentation.ndim) :]
    for dim in range(segmentation.ndim):
        dim_name = dim_names[dim]
        _centroids[dim_name] = _centroids.pop(f"{CENTROID_PROPERTY}-{dim}")

    return _centroids


def _concat_centroids(centroids, new_centroids):
    """Concatentate centroid dictionaries."""
    for key, values in new_centroids.items():
        if key not in centroids:
            centroids[key] = values
        else:
            centroids[key] = np.concatenate([centroids[key], values])
    return centroids


def segmentation_to_objects(
    segmentation: Union[np.ndarray, Generator],
    intensity_image: Optional[Union[np.ndarray, Generator]] = None,
    properties: Optional[Tuple[str]] = (),
    scale: Optional[Tuple[float]] = None,
    use_weighted_centroid: bool = True,
) -> list:
    """Convert segmentation to a set of btrack.PyTrackObject.

    Parameters
    ----------
    segmentation : np.ndarray or Generator
        Segmentation can be provided in several different formats. Arrays should
        be ordered as T(Z)YX.

    intensity_image : np.ndarray or Generator, optional
        Intensity image with same size as segmentation, to be used to calculate
        additional properties. See skimage.measure.regionprops for more info.

    properties : tuple of str, optional
        Properties passed to scikit-image regionprops. These additional
        properties are added as metadata to the btrack objects. See
        skimage.measure.regionprops for more info.

    scale : tuple
        A scale for each spatial dimension of the input segmentation. Defaults
        to one for all axes, and allows scaling for anisotropic imaging data.

    use_weighted_centroid : bool, default True
        If an intensity image has been provided, default to calculating the
        weighted centroid. See skimage.measure.regionprops for more info.

    Returns
    -------
    objects : list
        A list of btrack.PyTrackObjects
    """

    centroids = {}
    USE_INTENSITY = False
    USE_WEIGHTED = False

    logger.info("Localizing objects from segmentation...")

    # if we have an intensity image, add that here
    if intensity_image is not None:
        if type(intensity_image) != type(segmentation):
            raise TypeError(
                "Segmentation and intensity image must be the same type."
            )
        USE_INTENSITY = True
        USE_WEIGHTED = use_weighted_centroid and USE_INTENSITY

    if USE_INTENSITY:
        logger.info('Found intensity_image data')

    if USE_WEIGHTED:
        logger.info('Calculating weighted centroids using intensity_image')

    # we need to remove 'label' since this is a protected keyword for btrack
    # objects
    if 'label' in properties:
        logger.warning("Cannot use scikit-image label as a property.")
        del properties['label']

    if isinstance(segmentation, np.ndarray):

        if segmentation.ndim not in (3, 4):
            raise ValueError("Segmentation array must have 3 or 4 dims.")

        for frame in range(segmentation.shape[0]):
            seg = segmentation[frame, ...]
            intens = intensity_image[frame, ...] if USE_INTENSITY else None
            _centroids = _centroids_from_single_arr(
                seg,
                properties,
                frame,
                intensity_image=intens,
                scale=scale,
                use_weighted_centroid=USE_WEIGHTED,
            )

            # concatenate the centroids
            centroids = _concat_centroids(centroids, _centroids)

    elif inspect.isgeneratorfunction(segmentation) or isinstance(
        segmentation, Generator
    ):

        for frame, seg in enumerate(segmentation):
            intens = next(intensity_image) if USE_INTENSITY else None
            _centroids = _centroids_from_single_arr(
                seg,
                properties,
                frame,
                intensity_image=intens,
                scale=scale,
                use_weighted_centroid=USE_WEIGHTED,
            )

            # concatenate the centroids
            centroids = _concat_centroids(centroids, _centroids)

    else:

        raise TypeError(
            f"Segmentation of type {type(segmentation)} not accepted."
        )

    if not centroids:
        logger.warning("...Found no objects.")
        return []

    # now create the btrack objects
    objects = localizations_to_objects(centroids)
    n_frames = int(np.max(centroids["t"]) + 1)

    logger.info(f"...Found {len(objects)} objects in {n_frames} frames.")

    return objects
