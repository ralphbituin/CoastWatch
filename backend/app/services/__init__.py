"""Service layer.

Modules are kept independent and communicate through documented interfaces and
shared database schemas:

* ``camera/``   — frame acquisition (Phase 3)
* ``vision/``   — segmentation, waterline extraction, calibration (Phase 4)
* ``tides/``    — tide providers, CSV import, interpolation, phase (Phase 2)
* ``baseline/`` — expected shoreline vs. tide (Phase 5)
* ``anomaly/``  — residuals, velocity, scoring, detection (Phase 6)
* ``pipeline``  — orchestration of the above (Phase 4 onward)
"""
