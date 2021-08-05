"""
Default values and instances for the ObjectDetection interface.
"""
# - Using the same default factory for ObjectDetector as the Classifier
#   interface.
# - Providing classifier default here for convenience.
# noinspection PyProtectedMember
from smqtk_classifier._defaults import DFLT_CLASSIFIER_FACTORY  # noqa: F401, lgtm[py/unused-import]
from smqtk_detection.detection_element_factory import DetectionElementFactory
from smqtk_detection.impls.detection_element.memory \
    import MemoryDetectionElement

DFLT_DETECTION_FACTORY = DetectionElementFactory(
    MemoryDetectionElement, {}
)
