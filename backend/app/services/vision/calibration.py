"""Camera calibration (Phase 4).

Homography-based calibration using operator-identified reference points and
known real-world distances, giving shoreline displacement in meters when camera
geometry permits. Calibration profiles are versioned per camera; suspected
camera movement invalidates the active calibration.

Reference points use the shape stored on ``CameraCalibration.reference_points``::

    {"points": [{"pixel": [x, y], "world_m": [x, y]}, ...]}

At least four non-collinear points are required. ``compute_homography`` solves
the direct linear transform with Hartley (translate-to-centroid / unit-mean-
distance) normalization and reports the RMS reprojection error in pixels so
operators can judge fit quality — a poor fit is surfaced, never hidden.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# Minimum inlier ratio for the "camera has not moved" verdict.
_MIN_INLIER_RATIO = 0.5
# Minimum point correspondences to attempt a homography estimate.
_MIN_MATCHES = 15


class CalibrationError(ValueError):
    """Raised when a calibration cannot be computed honestly."""


@dataclass(slots=True)
class CalibrationProfile:
    camera_id: int
    version: int
    homography: list[list[float]] | None
    meters_per_pixel: float | None
    rms_error_px: float | None = None


def _points_from_reference(reference_points: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract pixel/world arrays from the stored reference_points structure."""
    if not isinstance(reference_points, dict):
        raise CalibrationError("reference_points must be an object with a 'points' list")
    points = reference_points.get("points")
    if not isinstance(points, list | tuple) or len(points) < 4:
        raise CalibrationError("at least 4 reference points are required")

    pixels, worlds = [], []
    for i, entry in enumerate(points):
        try:
            px = entry["pixel"]
            wd = entry["world_m"]
            px = (float(px[0]), float(px[1]))
            wd = (float(wd[0]), float(wd[1]))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise CalibrationError(f"reference point #{i} is malformed: {entry!r}") from exc
        pixels.append(px)
        worlds.append(wd)
    return np.asarray(pixels, dtype=np.float64), np.asarray(worlds, dtype=np.float64)


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    shifted = points - centroid
    distances = np.linalg.norm(shifted, axis=1)
    mean_distance = distances.mean()
    if mean_distance < 1e-12:
        raise CalibrationError("reference points are degenerate (all coincident)")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    normalized = (transform @ np.hstack([points, ones]).T).T
    return normalized[:, :2], transform


def compute_homography(
    reference_points: Any,
    *,
    camera_id: int = 0,
    version: int = 1,
    meters_per_pixel: float | None = None,
) -> CalibrationProfile:
    """Solve a pixel→world homography from operator reference points.

    Returns a profile carrying the 3×3 row-major homography and the RMS
    reprojection error in pixels. Raises :class:`CalibrationError` when the
    points are insufficient or degenerate.
    """
    pixels, worlds = _points_from_reference(reference_points)
    norm_px, t_px = _normalize_points(pixels)
    norm_wd, t_wd = _normalize_points(worlds)

    rows = []
    for (x, y), (X, Y) in zip(norm_px, norm_wd, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, x * X, y * X, X])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, x * Y, y * Y, Y])
    a = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(a)
    h = vh[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise CalibrationError("homography solve failed (degenerate configuration)")
    h = h / h[2, 2]
    # Denormalize back to original coordinates.
    homography = np.linalg.inv(t_wd) @ h @ t_px
    homography = homography / homography[2, 2]

    projected = apply_homography(homography, pixels)
    residuals = worlds - projected
    rms_error = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))

    return CalibrationProfile(
        camera_id=camera_id,
        version=version,
        homography=homography.tolist(),
        meters_per_pixel=meters_per_pixel,
        rms_error_px=rms_error,
    )


def apply_homography(homography: Any, points: Iterable[tuple[float, float]]) -> np.ndarray:
    """Map image-space ``points`` to world-space with a 3×3 homography."""
    h = np.asarray(homography, dtype=np.float64)
    if h.shape != (3, 3):
        raise CalibrationError("homography must be a 3×3 matrix")
    pts = np.asarray(list(points), dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    mapped = (h @ np.hstack([pts, ones]).T).T
    w = mapped[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise CalibrationError("homography maps a point to infinity")
    return mapped[:, :2] / w


def _gray(frame: Any) -> np.ndarray:
    if frame is None:
        raise CalibrationError("frame must not be None")
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def detect_camera_movement(
    previous_frame: Any,
    current_frame: Any,
    *,
    min_matches: int = _MIN_MATCHES,
    identity_drift_px: float = 2.0,
) -> bool:
    """Heuristic: has the camera moved between two frames?

    Matches ORB features — the static parts of the scene (sky, structures,
    posts) dominate the inliers — and estimates a frame-to-frame homography
    with RANSAC. A *static* camera yields a homography close to the identity;
    any re-aiming (rotation, tilt, shift, zoom) drifts it away from identity.
    Returns ``True`` when the camera may have moved — including when movement
    cannot be ruled out (too few features). ``False`` means a static scene
    with a near-identity homography and no evidence of re-aiming.

    Caveat: for a perfectly planar scene a rigid camera rotation is also a
    homography, so this check cannot distinguish "planar scene, camera moved"
    from a static planar scene if the drift is below ``identity_drift_px``;
    it is a tripwire for gross re-aiming, not a pixel-exact registration.
    """
    prev_gray, curr_gray = _gray(previous_frame), _gray(current_frame)
    orb = cv2.ORB_create(nfeatures=1000)  # type: ignore[attr-defined]
    kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)
    kp_curr, des_curr = orb.detectAndCompute(curr_gray, None)
    if des_prev is None or des_curr is None or len(kp_prev) < min_matches:
        logger.debug("camera movement check: too few features — cannot rule out movement")
        return True

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_prev, des_curr, k=2)
    good = [
        m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance
    ]
    if len(good) < min_matches:
        logger.debug("camera movement check: too few matches (%d)", len(good))
        return True

    src = np.array([kp_prev[m.queryIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array([kp_curr[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if homography is None or inlier_mask is None:
        return True

    inlier_ratio = float(inlier_mask.sum()) / float(len(inlier_mask))
    if inlier_ratio < _MIN_INLIER_RATIO:
        return True

    # Drift of the fitted homography relative to identity, measured on inliers.
    inlier_src = src[inlier_mask.ravel().astype(bool)]
    drift = cv2.perspectiveTransform(inlier_src, homography) - inlier_src
    mean_drift = float(np.sqrt(np.mean(np.sum(drift**2, axis=(1, 2)))))
    if mean_drift > identity_drift_px:
        logger.debug("camera movement check: mean identity drift %.2f px", mean_drift)
        return True
    return False
