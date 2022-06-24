Detection Interfaces
--------------------

Here we list and briefly describe the high level algorithm interfaces which SMQTK-Detection provides.
Some implementations will require additional dependencies that cannot be packaged with SMQTK-Detection.

Detection Element
+++++++++++++++++
Data structure used by detector interfaces to communicate inference
predictions.

.. autoclass:: smqtk_detection.interfaces.detection_element.DetectionElement
    :members:

ObjectDetector
++++++++++++++
This interface defines a method to generate object detections
(:class:`smqtk_detection.interfaces.detection_element.DetectionElement`) over a given
:class:`smqtk_dataprovider.interfaces.data_element.DataElement`.

.. autoclass:: smqtk_detection.interfaces.object_detector.ObjectDetector
   :members:

ImageMatrixObjectDetector
+++++++++++++++++++++++++
.. autoclass:: smqtk_detection.interfaces.object_detector.ImageMatrixObjectDetector
   :members:

DetectImageObjects
++++++++++++++++++
.. autoclass:: smqtk_detection.interfaces.detect_image_objects.DetectImageObjects
  :members:
