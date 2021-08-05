# noinspection PyUnresolvedReferences
# - Convenience pass-through from "private" sub-module.
from smqtk_classifier._defaults import DFLT_CLASSIFIER_FACTORY
from smqtk_detection._defaults import DFLT_DETECTION_FACTORY  # noqa: F401
from .object_detector import ObjectDetector, ImageMatrixObjectDetector  # noqa: F401
