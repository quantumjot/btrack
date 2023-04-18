from __future__ import annotations

import dataclasses
import inspect
import logging
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from skimage.measure import label, regionprops, regionprops_table

from btrack import btypes
from btrack.constants import Dimensionality

from .utils import localizations_to_objects

# get the logger instance
logger = logging.getLogger(__name__)


try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator, *args, **kwargs):
        logger.info("Try installing ``tqdm`` for progress bar rendering.")
        return iterator


def _is_unique(x: npt.NDArray) -> bool:
    """Check whether a segmentation is equivalent to the labeled version."""
    # check if image is uniquely labelled (necessary for regionprops)
    # return np.max(label(x)) == np.max(x)
    return np.array_equal(label(x), x)


def _nodes_from_single_arr(
    segmentation: npt.NDArray,
    properties: Tuple[str],
    frame: int,
    *,
    centroid_type: str = "centroid",
    intensity_image: Optional[npt.NDArray] = None,
    scale: Optional[Tuple[float]] = None,
    assign_class_ID: bool = False,
    extra_properties: Optional[Tuple[Callable]] = None,
) -> Dict[str, Any]:
    """Return the object centroids from a numpy array representing the
    image data."""

    if np.sum(segmentation) == 0:
        return {}

    if segmentation.ndim not in (Dimensionality.TWO, Dimensionality.THREE):
        raise ValueError("Segmentation array must have 3 or 4 dims.")

    labeled = (
        label(segmentation) if not _is_unique(segmentation) else segmentation
    )
    props = regionprops(
        labeled,
        intensity_image=intensity_image,
        extra_properties=extra_properties,
    )
    num_nodes = len(props)
    scale = tuple([1.0] * segmentation.ndim) if scale is None else scale

    if len(scale) != segmentation.ndim:
        raise ValueError(
            f"Scale dimensions do not match segmentation: {scale}."
        )

    centroids = list(
        zip(*[getattr(props[idx], centroid_type) for idx in range(num_nodes)])
    )[::-1]
    centroid_dims = ["x", "y", "z"][: segmentation.ndim]

    coords = {
        centroid_dims[dim]: np.asarray(centroids[dim]) * scale[::-1][dim]
        for dim in range(len(centroids))
    }

    nodes = {"t": [frame] * num_nodes}
    nodes.update(coords)

    extra_img_props = tuple(
        [str(fn.__name__) for fn in extra_properties]
        if extra_properties
        else []
    )
    img_props = properties + extra_img_props

    for img_prop in img_props:
        nodes[img_prop] = [
            getattr(props[idx], img_prop) for idx in range(num_nodes)
        ]

    if assign_class_ID:
        _class_id = regionprops_table(
            labeled,
            intensity_image=segmentation,
            properties=("max_intensity",),
        )
        nodes["class_id"] = _class_id["max_intensity"]

    return nodes


def _concat_nodes(
    nodes: Dict[str, Any], new_nodes: Dict[str, Any]
) -> Dict[str, Any]:
    """Concatentate centroid dictionaries."""
    for key, values in new_nodes.items():
        if key not in nodes:
            nodes[key] = values
        else:
            nodes[key] = np.concatenate([nodes[key], values])
    return nodes


@dataclasses.dataclass
class SegmentationContainer:
    """Container for segmentation data."""

    segmentation: Union[Generator, npt.NDArray]
    intensity_image: Optional[Union[Generator, npt.NDArray]] = None

    def __post_init__(self):
        self._is_generator = inspect.isgeneratorfunction(
            self.segmentation
        ) or isinstance(self.segmentation, Generator)

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._is_generator:
            seg = next(self.segmentation)
            intens = (
                next(self.intensity_image)
                if self.intensity_image is not None
                else None
            )
        elif self._iter < len(self):
            seg = np.asarray(self.segmentation[self._iter, ...])
            intens = (
                np.asarray(self.intensity_image[self._iter, ...])
                if self.intensity_image is not None
                else None
            )
        else:
            raise StopIteration

        data = (self._iter, seg, intens)
        self._iter += 1
        return data

    def __len__(self) -> int:
        if not self._is_generator:
            return self.segmentation.shape[0]
        return 0


def segmentation_to_objects(
    segmentation: Union[np.ndarray, Generator],
    *,
    intensity_image: Optional[Union[np.ndarray, Generator]] = None,
    properties: Optional[Tuple[str]] = (),
    extra_properties: Optional[Tuple[Callable]] = None,
    scale: Optional[Tuple[float]] = None,
    use_weighted_centroid: bool = True,
    assign_class_ID: bool = False,
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
    extra_properties : tuple of callable
        Callable functions to calculate additional properties.
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

    nodes: dict = {}
    logger.info("Localizing objects from segmentation...")

    centroid_type = (
        "centroid_weighted"
        if (use_weighted_centroid and intensity_image is not None)
        else "centroid"
    )

    # we need to remove 'label' since this is a protected keyword for btrack
    # objects
    if isinstance(properties, tuple) and "label" in properties:
        logger.warning("Cannot use scikit-image label as a property.")
        properties = set(properties)
        properties.remove("label")
        properties = tuple(properties)

    container = SegmentationContainer(segmentation, intensity_image)

    for frame, seg, intens in tqdm(container, total=len(container)):
        _nodes = _nodes_from_single_arr(
            seg,
            properties,
            frame,
            intensity_image=intens,
            scale=scale,
            centroid_type=centroid_type,
            assign_class_ID=assign_class_ID,
            extra_properties=extra_properties,
        )

        # concatenate the centroids
        nodes = _concat_nodes(nodes, _nodes)

    if not nodes:
        logger.warning("...Found no objects.")
        return []

    # now create the btrack objects
    objects = localizations_to_objects(nodes)
    n_frames = int(np.max(nodes["t"]) + 1)

    logger.info(f"...Found {len(objects)} objects in {n_frames} frames.")

    return objects
