"""The camera: where it stands relative to the screen, and how its lens bends straight lines.

A pose (yaw, pitch, roll, distance) gives a homography from page pixels to sensor pixels.
The distance is set so that one display pixel covers ``scale`` sensor pixels at the crop
centre. Between about 0.7 and 1.6, the screen's subpixel grid sits close to the sensor's
Nyquist limit, which is where moiré appears. Radial lens distortion is applied relative to
the full sensor, so crops near the edge bend more.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Pose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    scale: float  # sensor pixels per display pixel at the crop centre
    focal_px: float
    k1: float
    crop_offset: tuple[float, float]  # crop centre relative to the sensor centre (pixels)
    sensor_half_diag: float

    def as_dict(self) -> dict:
        return asdict(self)


def _uniform(rng: np.random.Generator, lo_hi) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def sample_pose(rng: np.random.Generator, geo: dict, crop: int) -> Pose:
    sw, sh = geo.get("sensor_size", [4000, 3000])
    max_off = np.array([sw - crop, sh - crop]) / 2
    return Pose(
        yaw_deg=float(rng.uniform(-geo["yaw_deg"], geo["yaw_deg"])),
        pitch_deg=float(rng.uniform(-geo["pitch_deg"], geo["pitch_deg"])),
        roll_deg=float(rng.uniform(-geo["roll_deg"], geo["roll_deg"])),
        scale=float(np.exp(rng.uniform(np.log(geo["scale"][0]), np.log(geo["scale"][1])))),  # log-uniform
        focal_px=float(geo.get("focal_px", 3000)),
        k1=_uniform(rng, geo.get("k1", [0.0, 0.0])),
        crop_offset=tuple(float(v) for v in rng.uniform(-max_off, max_off)),
        sensor_half_diag=float(np.hypot(sw, sh) / 2),
    )


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def page_to_crop(pose: Pose, center_uv: tuple[float, float], crop: int) -> np.ndarray:
    """Homography: page pixel (u, v) -> pinhole pixel in the crop, with ``center_uv`` at the crop centre."""
    f = pose.focal_px
    R = rotation(pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    d = f / pose.scale  # at this distance a unit on the plane spans ``scale`` pixels at the centre
    c = (crop - 1) / 2
    K = np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])
    # Plane point (X, Y, 0) -> camera R [X, Y, 0] + [0, 0, d]: the page point (0, 0) sits on the optical axis.
    Hplane = K @ np.column_stack([R[:, 0], R[:, 1], [0, 0, d]])
    Hplane /= Hplane[2, 2]
    T = np.array([[1, 0, -center_uv[0]], [0, 1, -center_uv[1]], [0, 0, 1.0]])
    return Hplane @ T


def crop_to_page_maps(pose: Pose, H_page_to_crop: np.ndarray, crop: int) -> tuple[np.ndarray, np.ndarray]:
    """For every sensor pixel of the crop, the page position it sees (lens distortion included)."""
    c = (crop - 1) / 2
    ys, xs = np.mgrid[0:crop, 0:crop].astype(np.float64)
    # Distorted sensor position -> ideal pinhole position (first-order inverse of r_d = r (1 + k1 r^2)).
    gx, gy = xs - c + pose.crop_offset[0], ys - c + pose.crop_offset[1]
    r2 = (gx**2 + gy**2) / pose.sensor_half_diag**2
    undist = 1.0 - pose.k1 * r2
    px = gx * undist - pose.crop_offset[0] + c
    py = gy * undist - pose.crop_offset[1] + c
    Hinv = np.linalg.inv(H_page_to_crop)
    den = Hinv[2, 0] * px + Hinv[2, 1] * py + Hinv[2, 2]
    u = (Hinv[0, 0] * px + Hinv[0, 1] * py + Hinv[0, 2]) / den
    v = (Hinv[1, 0] * px + Hinv[1, 1] * py + Hinv[1, 2]) / den
    return u.astype(np.float32), v.astype(np.float32)
