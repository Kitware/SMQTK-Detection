Pending Release Notes
=====================

Updates / New Features
----------------------

CI

* Updated CI unittests workflow to include codecov reporting.
  Reduced CodeCov report submission by skipping this step on scheduled runs.

Documentation

* Updated CONTRIBUTING.md to reference smqtk-core's CONTRIBUTING.md file.

Fixes
-----

Dependency Versions

* Updated the developer dependency and locked version of ipython to address a
  security vulnerability.

* Removed `jedi = "^0.17.2"` requirement since recent `ipython = "^7.17.3"`
  update appropriately addresses the dependency.
