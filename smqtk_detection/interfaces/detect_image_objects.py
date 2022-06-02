import abc
from typing import Iterable, Hashable, Dict, Tuple
import numpy as np

from smqtk_core import Configurable, Pluggable
from smqtk_image_io import AxisAlignedBoundingBox


class DetectImageObjects (Configurable, Pluggable):
    """
    Algorithm that generates object bounding boxes and classification maps for
    a set of input image matricies as ``numpy.ndarray`` type arrays.
    """

    @abc.abstractmethod
    def detect_objects(
      self,
      img_iter: Iterable[np.ndarray]
    ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
        """
        Generate paired bounding boxes and classification maps for detected
        objects in the given set of images.

        :param img_iter: Iterable of input images as numpy arrays.

        :return: Iterable of sets of paired bounding boxes and classification
            maps. Each set is the collection of detections for the
            corresponding input image.
        """

    def __call__(
      self,
      img_iter: Iterable[np.ndarray]
    ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
        """
        Calls `detect_objects() with the given iterable set of images.`
        """

        return self.detect_objects(img_iter)
