Pending Release Notes
=====================

Updates / New Features
----------------------

- Previously, malformed bboxes (``max_vertex < min_vertex``) returned by the internal model architecture of
  ``CenterNetVisdrone`` caused an exception to be raised by ``AxisAlignedBoundingBox`` during bbox construction. Now,
  ``reorder_malformed_bboxes`` is exposed as a configuration parameter to more appropriately handle behavior when
  malformed bboxes are encountered. If ``True``, bbox vertices will be sorted such that (``max_vertex >= min_vertex``),
  otherwise, (new default) the offending detection will be dropped from the list of detections completely. This new
  default is similar to previous behavior, but avoids hitting the exception such that the remaining well-formed
  detections can be returned.

Fixes
-----
