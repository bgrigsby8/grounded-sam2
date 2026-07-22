import numpy as np
import pytest

import pointcloud


FX, FY = 500.0, 500.0
CX, CY = 320.0, 240.0


def make_depth(h=480, w=640, value=1000):
    return np.full((h, w), value, dtype=np.uint16)


class TestDepthToPoints:
    def test_principal_point_backprojects_to_axis(self):
        """The pixel at the principal point lies on the camera's Z axis."""
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[int(CY), int(CX)] = 1500  # mm
        points = pointcloud.depth_to_points(depth, FX, FY, CX, CY)
        assert points.shape == (1, 3)
        np.testing.assert_allclose(points[0], [0.0, 0.0, 1500.0], atol=1e-3)

    def test_known_offset_pixel(self):
        """X = (u - cx) * Z / fx: 100px right at 1m depth with fx=500 -> X=200mm."""
        depth = np.zeros((480, 640), dtype=np.uint16)
        u, v, z = int(CX) + 100, int(CY) + 50, 1000
        depth[v, u] = z
        points = pointcloud.depth_to_points(depth, FX, FY, CX, CY)
        np.testing.assert_allclose(points[0], [200.0, 100.0, 1000.0], atol=1e-3)

    def test_zero_depth_dropped(self):
        depth = make_depth(value=0)
        points = pointcloud.depth_to_points(depth, FX, FY, CX, CY)
        assert points.shape == (0, 3)

    def test_mask_selects_subset(self):
        depth = make_depth(value=800)
        mask = np.zeros(depth.shape, dtype=bool)
        mask[10:20, 30:40] = True
        points = pointcloud.depth_to_points(depth, FX, FY, CX, CY, mask=mask)
        assert points.shape == (100, 3)
        assert (points[:, 2] == 800).all()

    def test_mask_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="mask shape"):
            pointcloud.depth_to_points(
                make_depth(), FX, FY, CX, CY, mask=np.ones((10, 10), dtype=bool)
            )

    def test_bad_intrinsics_raise(self):
        with pytest.raises(ValueError, match="intrinsics"):
            pointcloud.depth_to_points(make_depth(), 0.0, FY, CX, CY)


class TestDownsample:
    def test_no_op_when_small(self):
        pts = np.random.default_rng(1).random((100, 3)).astype(np.float32)
        out = pointcloud.downsample_points(pts, 1000)
        assert out is pts

    def test_reduces_to_max(self):
        pts = np.random.default_rng(1).random((60_000, 3)).astype(np.float32)
        out = pointcloud.downsample_points(pts, 50_000)
        assert out.shape == (50_000, 3)

    def test_deterministic(self):
        pts = np.random.default_rng(1).random((1000, 3)).astype(np.float32)
        a = pointcloud.downsample_points(pts, 500)
        b = pointcloud.downsample_points(pts, 500)
        np.testing.assert_array_equal(a, b)


class TestPCD:
    def test_roundtrip(self):
        pts = np.array([[1.0, 2.0, 3.0], [-4.5, 0.0, 9.25]], dtype=np.float32)
        data = pointcloud.encode_pcd(pts)
        out = pointcloud.decode_pcd(data)
        np.testing.assert_allclose(out, pts)

    def test_scale_mm_to_m(self):
        pts = np.array([[1000.0, 2000.0, 500.0]], dtype=np.float32)
        out = pointcloud.decode_pcd(pointcloud.encode_pcd(pts, scale=0.001))
        np.testing.assert_allclose(out, [[1.0, 2.0, 0.5]])

    def test_header_fields(self):
        data = pointcloud.encode_pcd(np.zeros((7, 3), dtype=np.float32))
        header = data.split(b"DATA binary\n")[0].decode("ascii")
        assert "FIELDS x y z" in header
        assert "POINTS 7" in header
        assert "WIDTH 7" in header

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            pointcloud.encode_pcd(np.zeros((5, 4)))


class TestAxisAlignedBox:
    def test_center_and_dims(self):
        pts = np.array([[0, 0, 0], [10, 20, 30]], dtype=np.float32)
        center, dims = pointcloud.axis_aligned_box(pts)
        np.testing.assert_allclose(center, [5, 10, 15])
        np.testing.assert_allclose(dims, [10, 20, 30])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            pointcloud.axis_aligned_box(np.zeros((0, 3)))


class TestViamRawDepthDecode:
    def encode(self, arr: np.ndarray) -> bytes:
        h, w = arr.shape
        return (
            pointcloud.VIAM_RAW_DEPTH_MAGIC
            + w.to_bytes(8, "big")
            + h.to_bytes(8, "big")
            + arr.astype(">u2").tobytes()
        )

    def test_roundtrip(self):
        arr = np.arange(12, dtype=np.uint16).reshape(3, 4) * 100
        out = pointcloud.decode_viam_raw_depth(self.encode(arr))
        np.testing.assert_array_equal(out, arr)

    def test_bad_magic_raises(self):
        with pytest.raises(ValueError, match="DEPTHMAP"):
            pointcloud.decode_viam_raw_depth(b"NOTDEPTH" + b"\x00" * 32)

    def test_truncated_raises(self):
        arr = np.ones((4, 4), dtype=np.uint16)
        with pytest.raises(ValueError, match="truncated"):
            pointcloud.decode_viam_raw_depth(self.encode(arr)[:-3])


class TestEndToEnd:
    def test_synthetic_object_lands_where_expected(self):
        """A flat 100x100px patch at 2m depth, centered on the principal
        point, should produce a cloud centered on (0, 0, 2000)mm."""
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[190:290, 270:370] = 2000
        mask = depth > 0

        points = pointcloud.depth_to_points(depth, FX, FY, CX, CY, mask=mask)
        assert points.shape == (10_000, 3)
        center, dims = pointcloud.axis_aligned_box(points)
        # patch spans +/-50px around the principal point -> +/-200mm at 2m
        np.testing.assert_allclose(center[2], 2000.0)
        assert abs(center[0]) < 5.0 and abs(center[1]) < 5.0
        np.testing.assert_allclose(dims[:2], [396.0, 396.0], atol=5.0)

        decoded = pointcloud.decode_pcd(pointcloud.encode_pcd(points, scale=0.001))
        np.testing.assert_allclose(decoded[:, 2], 2.0, atol=1e-6)
