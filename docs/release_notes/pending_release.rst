Pending Release Notes
=====================

Updates / New Features
----------------------

CI

* Updated CI unittests workflow to include codecov reporting.
  Reduced CodeCov report submission by skipping this step on scheduled runs.

Documentation

* Updated CONTRIBUTING.md to reference smqtk-core's CONTRIBUTING.md file.

Detect Image Objects

* Updated the `ResNetFRCNN` to return as its class labels the label strings
  instead of integers, reducing the burden of users from having to repeatedly
  find and allocate the appropriate int-to-label map.

* Update `CenterNetVisdrone` to lazy load its model on first inference.

* Update `CenterNetVisdrone` to return VisDrone class labels instead of
  integers.

Unit Tests

* Added unit tests for the `CenterNetVisdrone` plugin.

Utilities

* Remove duplicate implementation of `AxisAlignedBoundingBox`, using the tested
  version that comes from the ``smqtk_image_io`` package.

Fixes
-----

Detect Image Object

* Fixed batched operation memory usage in `ResNetFRCNN` by loading only current
  batch into computation device memory. Previously all images were loaded at
  once.

* Fixed device mapping when loading certain background architectures for
  `CenterNetVisdrone`.

Dependency Versions

* Updated the developer dependency and locked version of ipython to address a
  security vulnerability.

* Removed `jedi = "^0.17.2"` requirement since recent `ipython = "^7.17.3"`
  update appropriately addresses the dependency.
