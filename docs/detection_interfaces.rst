Detection Interfaces
--------------------

Here we list and briefly describe the high level algorithm interfaces which SMQTK-Detection provides.
Some implementations will require additional dependencies that cannot be packaged with SMQTK-Detection.

ObjectDetector
++++++++++++++
This interface defines a method to generate object detections
(:class:`smqtk_detection.interfaces.detection_element.DetectionElement`) over a given
:class:`smqtk_dataprovider.interfaces.data_element.DataElement`.

.. autoclass:: smqtk_detection.interfaces.object_detector.ObjectDetector
   :members:

Detection Element
+++++++++++++++++
Data structure used by Detector

.. autoclass:: smqtk_detection.interfaces.detection_element.DetectionElement
    :members:

ImageMatrixObjectDetector
+++++++++++++++++++++++++
.. autoclass:: smqtk_detection.interfaces.object_detector.ImageMatrixObjectDetector
   :members:
