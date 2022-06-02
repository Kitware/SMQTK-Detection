from random import randrange
from typing import Iterable, Tuple, Dict, Hashable, Sequence

import numpy as np
from smqtk_image_io import AxisAlignedBoundingBox

from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects


class RandomDetector(DetectImageObjects):
    """
    Example implementation of the `DetectImageObjects` interface. An instance
    of this class acts as a functor to generate paired bounding boxes and
    classification maps for objects detected in a set of images.
    """

    def detect_objects(
        self,
        img_iter: Iterable[np.ndarray]
    ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
        """
        Return random set of detections for each image in the input set.
        """

        dets = []

        for img in img_iter:
            img_h = img.shape[0]
            img_w = img.shape[1]

            num_dets = randrange(10)

            dets.append(
                [(
                    self._gen_random_bbox(img_w, img_h),
                    self._gen_random_class_map([0, 1, 2])
                ) for _ in range(num_dets)]
            )

        return dets

    def _gen_random_bbox(self, img_w: int, img_h: int) -> AxisAlignedBoundingBox:
        """
        Creates `AxisAlignedBoundingBox` object with random vertices within
        passed image size.
        """

        min_vertex = [randrange(int(img_w/2)), randrange(int(img_h/2))]
        max_vertex = [randrange(int(img_w/2), img_w), randrange(int(img_h/2), img_h)]

        return AxisAlignedBoundingBox(min_vertex, max_vertex)

    def _gen_random_class_map(self, classes: Sequence) -> Dict[Hashable, float]:
        """
        Creates dictionary of random classification scores for the list of
        input classes.
        """

        scores = np.random.rand(len(classes))
        scores = scores / scores.sum()

        d = {}
        for i, c in enumerate(classes):
            d[c] = scores[i]

        return d

    def get_config(self) -> dict:
        return {}
