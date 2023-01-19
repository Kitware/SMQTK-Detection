Pending Release Notes
=====================

Updates / New Features
----------------------

Build

* Remove explicit cap to the python version.

Dependencies

* Updated python minimum requirement to 3.7 (up from 3.6). Thin involved a
  number of updates and bifurcations of abstract requirements, an update to
  pinned versions for development/CI, and expansion of CI to cover python
  versions 3.10 and 3.11 (latest current release).

Fixes
-----

Docs

* Fix erroneous references to previous monorepo.

* Fixed ``sphinx_server.py`` to reference correct directories.
