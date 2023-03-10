from __future__ import annotations

import inspect
import logging
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
from skimage.measure import label, regionprops_table

from btrack import btypes
from btrack.constants import Dimensionality

from .utils import localizations_to_objects

# get the logger instance
logger = logging.getLogger(__name__)


def _centroids_from_single_arr(
    segmentation: Union[np.ndarray, Generator],
    properties: Tuple[str],
    frame: int,
    intensity_image: Optional[np.ndarray] = None,
    scale: Optional[Tuple[float]] = None,
    *,
    use_weighted_centroid: bool = False,
    assign_class_ID: bool = False,
) -> np.ndarray:
    """Return the object centroids from a numpy array representing the
    image data."""

    if np.sum(segmentation) == 0:
        return {}

    def _is_unique(x: np.ndarray) -> bool:
        # check if image is uniquely labelled (necessary for regionprops)
        return np.max(label(x)) == np.max(x)

    if use_weighted_centroid and intensity_image is not None:
        CENTROID_PROPERTY = "weighted_centroid"
    else:
        CENTROID_PROPERTY = "centroid"

    if CENTROID_PROPERTY not in properties:
        properties = (CENTROID_PROPERTY, *properties)

    # if class id is specified then extract that property first
    if assign_class_ID:

        # ensure regionprops can properly read label image
        labeled = label(segmentation)

        # pull class_ID from segments using pixel intensity
        _class_ID_centroids = regionprops_table(
            labeled,
            intensity_image=segmentation,
            properties=("max_intensity",),
        )

        # rename class_ID column and remove keyword from properties
        _class_ID_centroids["class_id"] = _class_ID_centroids.pop(
            "max_intensity"
        )

        # run regionprops to record other intensity image properties
        _centroids = regionprops_table(
            labeled,
            intensity_image=intensity_image,
            properties=properties,
        )

        # merge centroids with class ID centroids
        _centroids.update(_class_ID_centroids)

        assert "class_id" in _centroids, _centroids.keys()

    else:
        # check to see whether the segmentation is unique
        labeled = (
            label(segmentation)
            if not _is_unique(segmentation)
            else segmentation
        )

        _centroids = regionprops_table(
            labeled,
            intensity_image=intensity_image,
            properties=properties,
        )

    # add time to the array
    _centroids["t"] = np.full(
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


def segmentation_to_objects(  # noqa: PLR0913
    segmentation: Union[np.ndarray, Generator],
    intensity_image: Optional[Union[np.ndarray, Generator]] = None,
    properties: Optional[Tuple[str]] = (),
    scale: Optional[Tuple[float]] = None,
    use_weighted_centroid: bool = True,  # noqa: FBT001 FBT002
    assign_class_ID: bool = False,  # noqa: FBT001 FBT002
) -> List[btypes.PyTrackObject]:
    """Convert segmentation to a set of trackable objects.

    Parameters
    ----------
    segmentation : np.ndarray, dask.array.core.Array or Generator
        Segmentation can be provided in several different formats. Arrays should
        be ordered as T(Z)YX.
    intensity_image : np.ndarray, dask.array.core.Array or Generator, optional
        Intensity image with same size as segmentation, to be used to calculate
        additional properties. See `skimage.measure.regionprops` for more info.
    properties : tuple of str, optional
        Properties passed to scikit-image regionprops. These additional
        properties are added as metadata to the btrack objects.
        See `skimage.measure.regionprops` for more info.
    scale : tuple
        A scale for each spatial dimension of the input segmentation. Defaults
        to one for all axes, and allows scaling for anisotropic imaging data.
    use_weighted_centroid : bool, default True
        If an intensity image has been provided, default to calculating the
        weighted centroid. See `skimage.measure.regionprops` for more info.
    assign_class_ID : bool, default False
        If specified, assign a class label for each individual object based on
        the pixel intensity found in the mask. Requires semantic segmentation,
        i.e. object type 1 will have pixel value 1.

    Returns
    -------
    objects : list
        A list of :py:meth:`btrack.btypes.PyTrackObject` trackable objects.

    Examples
    --------
    >>> objects = btrack.utils.segmentation_to_objects(
    ...   segmentation,
    ...   properties=('area', ),
    ...   scale=(1., 1.),
    ...   assign_class_ID=True,
    ... )
    """

    centroids: dict = {}
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
        logger.info("Found intensity_image data")

    if USE_WEIGHTED:
        logger.info("Calculating weighted centroids using intensity_image")

    # we need to remove 'label' since this is a protected keyword for btrack
    # objects
    if isinstance(properties, tuple) and "label" in properties:
        logger.warning("Cannot use scikit-image label as a property.")
        properties = set(properties)
        properties.remove("label")
        properties = tuple(properties)

    if inspect.isgeneratorfunction(segmentation) or isinstance(
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
                assign_class_ID=assign_class_ID,
            )

            # concatenate the centroids
            centroids = _concat_centroids(centroids, _centroids)

    else:

        if segmentation.ndim not in (
            Dimensionality.THREE,
            Dimensionality.FOUR,
        ):
            raise ValueError("Segmentation array must have 3 or 4 dims.")

        for frame in range(segmentation.shape[0]):
            # try to cast to numpy array, should work for dask arrays and implicitly
            # call the `.compute()` method
            seg = np.asarray(segmentation[frame, ...])
            intens = (
                np.asarray(intensity_image[frame, ...])
                if USE_INTENSITY
                else None
            )
            _centroids = _centroids_from_single_arr(
                seg,
                properties,
                frame,
                intensity_image=intens,
                scale=scale,
                use_weighted_centroid=USE_WEIGHTED,
                assign_class_ID=assign_class_ID,
            )

            # concatenate the centroids
            centroids = _concat_centroids(centroids, _centroids)

    if not centroids:
        logger.warning("...Found no objects.")
        return []

    # now create the btrack objects
    objects = localizations_to_objects(centroids)
    n_frames = int(np.max(centroids["t"]) + 1)

    logger.info(f"...Found {len(objects)} objects in {n_frames} frames.")

    return objects
