Pending Release Notes
=====================

Updates / New Features
----------------------

CI / CD

* Added a dependency review workflow to improve tracking and resolution of
  dependabot alerts.

* Resolved cache size issues by splitting the unit test workflow into separate
  *core* and *extras* jobs, with the *extras* job depending on *core*.

Fixes
-----
