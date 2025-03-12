Pending Release Notes
=====================

Updates / New Features
----------------------

CI/CD

* Updated Github Workflows from ``cache@v2`` to ``cache@v4``.

* To resolve issues with public forks lacking access to the Codecov
  token, we've included it directly in ``codecov.yml``. The file also
  documents the rationale for this security exception.

Fixes
-----

* Fixed a bug in ``CenterNetVisdrone`` that prevented running on MPS device.
