from __future__ import annotations

import dataclasses
import logging
from collections.abc import Generator
from multiprocessing.pool import Pool
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
from skimage.measure import label, regionprops, regionprops_table
from tqdm import tqdm

from btrack import btypes
from btrack.constants import Dimensionality

from .utils import localizations_to_objects

# get the logger instance
logger = logging.getLogger(__name__)


def _is_unique(x: npt.NDArray) -> bool:
    """Check whether a segmentation is equivalent to the labeled version."""
    return np.array_equal(label(x), x)


def _concat_nodes(
    nodes: dict[str, npt.NDArray], new_nodes: dict[str, npt.NDArray]
) -> dict[str, npt.NDArray]:
    """Concatentate centroid dictionaries."""
    for key, values in new_nodes.items():
        nodes[key] = (
            np.concatenate([nodes[key], values]) if key in nodes else values
        )
    return nodes


@dataclasses.dataclass
class SegmentationContainer:
    """Container for segmentation data."""

    segmentation: Union[Generator, npt.NDArray]
    intensity_image: Optional[Union[Generator, npt.NDArray]] = None

    def __post_init__(self) -> None:
        self._is_generator = isinstance(self.segmentation, Generator)
        self._next = (
            self._next_generator if self._is_generator else self._next_array
        )

    def _next_generator(self) -> tuple[npt.NDArray, Optional[npt.NDArray]]:
        """__next__ method for a generator input."""
        seg = next(self.segmentation)
        intens = (
            next(self.intensity_image)
            if self.intensity_image is not None
            else None
        )
        return seg, intens

    def _next_array(self) -> tuple[npt.NDArray, Optional[npt.NDArray]]:
        """__next__ method for an array-like input."""
        if self._iter >= len(self):
            raise StopIteration
        seg = np.asarray(self.segmentation[self._iter, ...])
        intens = (
            np.asarray(self.intensity_image[self._iter, ...])
            if self.intensity_image is not None
            else None
        )
        return seg, intens

    def __iter__(self) -> SegmentationContainer:
        self._iter = 0
        return self

    def __next__(self) -> tuple[int, npt.NDArray, Optional[npt.NDArray]]:
        seg, intens = self._next()
        data = (self._iter, seg, intens)
        self._iter += 1
        return data

    def __len__(self) -> int:
        return 0 if self._is_generator else self.segmentation.shape[0]


@dataclasses.dataclass
class NodeProcessor:
    """Processor to extract nodes from a segmentation image."""

    properties: tuple[str]
    centroid_type: str = "centroid"
    intensity_image: Optional[npt.NDArray] = None
    scale: Optional[tuple[float]] = None
    assign_class_ID: bool = False  # noqa: N815
    extra_properties: Optional[tuple[Callable]] = None

    @property
    def img_props(self) -> list[str]:
        # need to infer the name of the function provided
        extra_img_props = tuple(
            [str(fn.__name__) for fn in self.extra_properties]
            if self.extra_properties
            else []
        )
        return self.properties + extra_img_props

    def __call__(
        self, data: tuple[int, npt.NDArray, Optional[npt.NDArray]]
    ) -> dict[str, npt.NDArray]:
        """Return the object centroids from a numpy array representing the
        image data."""

        frame, segmentation, intensity_image = data

        if np.sum(segmentation) == 0:
            return {}

        if segmentation.ndim not in (Dimensionality.TWO, Dimensionality.THREE):
            raise ValueError("Segmentation array must have 3 or 4 dims.")

        labeled = (
            segmentation if _is_unique(segmentation) else label(segmentation)
        )
        props = regionprops(
            labeled,
            intensity_image=intensity_image,
            extra_properties=self.extra_properties,
        )
        num_nodes = len(props)
        scale = (
            tuple([1.0] * segmentation.ndim)
            if self.scale is None
            else self.scale
        )

        if len(scale) != segmentation.ndim:
            raise ValueError(
                f"Scale dimensions do not match segmentation: {scale}."
            )

        centroids = list(
            zip(
                *[
                    getattr(props[idx], self.centroid_type)
                    for idx in range(num_nodes)
                ]
            )
        )[::-1]
        centroid_dims = ["x", "y", "z"][: segmentation.ndim]

        coords = {
            centroid_dims[dim]: np.asarray(centroids[dim]) * scale[::-1][dim]
            for dim in range(len(centroids))
        }

        nodes = {"t": [frame] * num_nodes} | coords
        for img_prop in self.img_props:
            nodes[img_prop] = [
                getattr(props[idx], img_prop) for idx in range(num_nodes)
            ]

        if self.assign_class_ID:
            _class_id = regionprops_table(
                labeled,
                intensity_image=segmentation,
                properties=("max_intensity",),
            )
            nodes["class_id"] = _class_id["max_intensity"]

        return nodes


def segmentation_to_objects(  # noqa: PLR0913
    segmentation: Union[npt.NDArray, Generator],
    *,
    intensity_image: Optional[Union[npt.NDArray, Generator]] = None,
    properties: Optional[tuple[str]] = None,
    extra_properties: Optional[tuple[Callable]] = None,
    scale: Optional[tuple[float]] = None,
    use_weighted_centroid: bool = True,
    assign_class_ID: bool = False,
    num_workers: int = 1,
) -> list[btypes.PyTrackObject]:
    """Convert segmentation to a set of trackable objects.

    Parameters
    ----------
    segmentation : npt.NDArray, dask.array.core.Array or Generator
        Segmentation can be provided in several different formats. Arrays should
        be ordered as T(Z)YX.
    intensity_image : npt.NDArray, dask.array.core.Array or Generator, optional
        Intensity image with same size as segmentation, to be used to calculate
        additional properties. See `skimage.measure.regionprops` for more info.
    properties : tuple of str, optional
        Properties passed to scikit-image regionprops. These additional
        properties are added as metadata to the btrack objects.
        See `skimage.measure.regionprops` for more info.
    extra_properties : tuple of callable, optional
        Callable functions to calculate additional properties from the
        segmentation and intensity image data. See `skimage.measure.regionprops`
        for more info.
    scale : tuple, optional
        A scale for each spatial dimension of the input segmentation. Defaults
        to one for all axes, and allows scaling for anisotropic imaging data.
    use_weighted_centroid : bool, default True
        If an intensity image has been provided, default to calculating the
        weighted centroid. See `skimage.measure.regionprops` for more info.
    assign_class_ID : bool, default False
        If specified, assign a class label for each individual object based on
        the pixel intensity found in the mask. Requires semantic segmentation,
        i.e. object type 1 will have pixel value 1.
    num_workers : int
        Number of workers to use while processing the image data.

    Returns
    -------
    objects : list
        A list of :py:meth:`btrack.btypes.PyTrackObject` trackable objects.


    Notes
    -----
    If `tqdm` is installed, a progress bar will be provided.

    Examples
    --------
    >>> objects = btrack.utils.segmentation_to_objects(
    ...   segmentation,
    ...   properties=('area', ),
    ...   scale=(1., 1.),
    ...   assign_class_ID=True,
    ...   num_workers=4,
    ... )

    It's also possible to provide custom analysis functions :

    >>> def foo(_mask: npt.NDArray) -> float:
    ...     return np.sum(_mask)

    that can be passed to :py:func:`btrack.utils.segmentation_to_objects` :

    >>> objects = btrack.utils.segmentation_to_objects(
    ...   segmentation,
    ...   extra_properties=(foo, ),
    ...   num_workers=1,
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
        logger.warning("Cannot use `scikit-image` `label` as a property.")
        properties = set(properties)
        properties.remove("label")
        properties = tuple(properties)

    processor = NodeProcessor(
        properties=properties,
        scale=scale,
        centroid_type=centroid_type,
        assign_class_ID=assign_class_ID,
        extra_properties=extra_properties,
    )

    container = SegmentationContainer(segmentation, intensity_image)

    if extra_properties:
        logger.warning(
            "Cannot use multiprocessing when `extra_properties` are defined."
        )
        num_workers = 1

    if num_workers <= 1:
        for data in tqdm(container, total=len(container), position=0):
            _nodes = processor(data)
            nodes = _concat_nodes(nodes, _nodes)
    else:
        logger.info(f"Processing using {num_workers} workers.")
        with Pool(processes=num_workers) as pool:
            result = list(
                tqdm(
                    pool.imap(processor, container),
                    total=len(container),
                    position=0,
                )
            )

        for _nodes in result:
            nodes = _concat_nodes(nodes, _nodes)

    if not nodes:
        logger.warning("...Found no objects.")
        return []

    # now create the btrack objects
    objects = localizations_to_objects(nodes)
    n_frames = int(max(nodes["t"]) + 1)

    logger.info(f"...Found {len(objects)} objects in {n_frames} frames.")

    return objects
