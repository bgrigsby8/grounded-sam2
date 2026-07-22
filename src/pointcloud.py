"""Depth back-projection and PCD encoding. No Viam SDK imports.

Unit conventions (matching Viam RDK):
- Depth images from Viam cameras are uint16 millimeters.
- PCD bytes handed back through the vision API are in METERS (the PCL/PCD
  convention; RDK converts to its internal mm representation on read).
- Geometry (bounding box center/dims) is reported in MILLIMETERS.

All functions here are unit-agnostic — points come out in whatever unit the
depth map is in; the caller scales when encoding.
"""

from typing import Optional, Tuple

import numpy as np

VIAM_RAW_DEPTH_MAGIC = b"DEPTHMAP"


def decode_viam_raw_depth(data: bytes) -> np.ndarray:
    """Decode Viam's image/vnd.viam.dep format into an (H, W) uint16 array.

    Layout: 8-byte magic "DEPTHMAP", big-endian uint64 width, big-endian
    uint64 height, then H*W big-endian uint16 depth values (row-major).
    """
    if len(data) < 24 or data[:8] != VIAM_RAW_DEPTH_MAGIC:
        raise ValueError("not a Viam raw depth image (missing DEPTHMAP header)")
    width = int.from_bytes(data[8:16], "big")
    height = int.from_bytes(data[16:24], "big")
    expected = width * height * 2
    body = data[24 : 24 + expected]
    if len(body) != expected:
        raise ValueError(
            f"depth image truncated: expected {expected} bytes for {width}x{height}, got {len(body)}"
        )
    return np.frombuffer(body, dtype=">u2").reshape(height, width).astype(np.uint16)


def depth_to_points(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Back-project a depth image to 3D points in the camera frame.

    Pinhole model: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = depth.
    Zero-depth pixels (no reading) are dropped. Output units match `depth`.

    Returns an (N, 3) float32 array.
    """
    if fx <= 0 or fy <= 0:
        raise ValueError(f"invalid intrinsics: fx={fx}, fy={fy}")
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got shape {depth.shape}")

    valid = depth > 0
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match depth shape {depth.shape}"
            )
        valid &= mask

    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float32)
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    return np.column_stack((x, y, z))


def downsample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Uniform random subsample to at most max_points (deterministic)."""
    n = points.shape[0]
    if n <= max_points:
        return points
    idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
    idx.sort()
    return points[idx]


def encode_pcd(points: np.ndarray, scale: float = 1.0) -> bytes:
    """Encode an (N, 3) point array as a binary PCD file (x y z float32).

    `scale` is applied to all coordinates — pass 0.001 to emit meters from
    millimeter-unit points.
    """
    pts = (np.asarray(points, dtype=np.float32) * scale).astype("<f4")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got shape {np.asarray(points).shape}")
    n = pts.shape[0]
    header = (
        "VERSION .7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    return header.encode("ascii") + pts.tobytes()


def decode_pcd(data: bytes) -> np.ndarray:
    """Minimal binary/ascii x-y-z PCD decoder (for tests/round-tripping)."""
    end = data.index(b"DATA")
    newline = data.index(b"\n", end)
    header = data[:newline].decode("ascii")
    fields = {}
    for line in header.splitlines():
        parts = line.split()
        if parts:
            fields[parts[0]] = parts[1:]
    n = int(fields["POINTS"][0])
    mode = fields["DATA"][0]
    body = data[newline + 1 :]
    if mode == "binary":
        return np.frombuffer(body[: n * 12], dtype="<f4").reshape(n, 3).copy()
    rows = [list(map(float, line.split())) for line in body.decode("ascii").split("\n") if line.strip()]
    return np.array(rows, dtype=np.float32).reshape(n, 3)


def axis_aligned_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of an (N, 3) point set.

    Returns (center, dims), each shape (3,), in the same units as `points`.
    """
    pts = np.asarray(points)
    if pts.size == 0:
        raise ValueError("cannot compute bounding box of empty point set")
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return (mins + maxs) / 2.0, maxs - mins
