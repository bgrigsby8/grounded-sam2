"""Service-level tests with a fake camera and stubbed pipeline (no model
downloads, no robot). Mirrors the lifecycle viam-server drives:
validate_config -> new(config, dependencies) -> RPC methods.
"""

import base64
import io

import numpy as np
import pytest
from google.protobuf.struct_pb2 import Struct
from PIL import Image
from viam.components.camera import Camera
from viam.media.utils.pil import pil_to_viam_image
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.component.camera import IntrinsicParameters

import pipeline as gs_pipeline
import pointcloud as gs_pointcloud
from models.vision import Vision

W, H = 64, 48
FX, FY, CX, CY = 100.0, 100.0, 32.0, 24.0


def make_config(attrs: dict) -> ComponentConfig:
    s = Struct()
    s.update(attrs)
    return ComponentConfig(name="gs2", attributes=s)


VALID_ATTRS = {"camera_name": "cam", "default_queries": ["bolt"]}


def encode_depth(arr: np.ndarray) -> bytes:
    h, w = arr.shape
    return (
        gs_pointcloud.VIAM_RAW_DEPTH_MAGIC
        + w.to_bytes(8, "big")
        + h.to_bytes(8, "big")
        + arr.astype(">u2").tobytes()
    )


class FakeCamera(Camera):
    def __init__(self, name: str, with_depth: bool = True, with_intrinsics: bool = True):
        super().__init__(name)
        self.with_depth = with_depth
        self.with_intrinsics = with_intrinsics
        self.rgb = Image.new("RGB", (W, H), (90, 90, 90))
        self.depth = np.zeros((H, W), dtype=np.uint16)
        self.depth[10:30, 20:40] = 1000  # 20x20 patch at 1m

    async def get_image(self, mime_type="", *, extra=None, timeout=None, **kwargs):
        return pil_to_viam_image(self.rgb, CameraMimeType.JPEG)

    async def get_images(self, *, filter_source_names=None, extra=None, timeout=None, **kwargs):
        images = [
            NamedImage(
                "color",
                pil_to_viam_image(self.rgb, CameraMimeType.JPEG).data,
                CameraMimeType.JPEG,
            )
        ]
        if self.with_depth:
            images.append(
                NamedImage("depth", encode_depth(self.depth), CameraMimeType.VIAM_RAW_DEPTH)
            )
        from viam.proto.common import ResponseMetadata

        return images, ResponseMetadata()

    async def get_point_cloud(self, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError

    async def get_properties(self, *, timeout=None, **kwargs):
        intrinsics = (
            IntrinsicParameters(
                width_px=W, height_px=H, focal_x_px=FX, focal_y_px=FY,
                center_x_px=CX, center_y_px=CY,
            )
            if self.with_intrinsics
            else IntrinsicParameters()
        )
        return Camera.Properties(
            supports_pcd=False, intrinsic_parameters=intrinsics
        )

    async def get_geometries(self, *, extra=None, timeout=None, **kwargs):
        return []


class StubPipeline:
    """Returns one fixed segment/detection matching FakeCamera's depth patch."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def _mask(self):
        mask = np.zeros((H, W), dtype=bool)
        mask[10:30, 20:40] = True
        return mask

    def detect(self, image, queries, **kw):
        self.calls.append(("detect", queries, kw))
        return [gs_pipeline.Detection2D(queries[0], 0.9, (20.0, 10.0, 40.0, 30.0))]

    def segment(self, image, queries, **kw):
        self.calls.append(("segment", queries, kw))
        return [
            gs_pipeline.Segment2D(queries[0], 0.9, (20.0, 10.0, 40.0, 30.0), self._mask())
        ]

    def unload(self):
        pass


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(gs_pipeline, "GroundedSam2Pipeline", StubPipeline)
    import models.vision as mv

    monkeypatch.setattr(mv.gs_pipeline, "GroundedSam2Pipeline", StubPipeline)
    camera = FakeCamera("cam")
    deps = {Camera.get_resource_name("cam"): camera}
    svc = Vision.new(make_config(VALID_ATTRS), deps)
    return svc, camera


class TestLifecycle:
    def test_new_applies_config(self, service):
        """Regression: EasyResource.new does not call reconfigure; ours must,
        otherwise every RPC dies with an AssertionError on _config."""
        svc, _ = service
        assert svc._config is not None
        assert svc._config.camera_name == "cam"
        assert svc._camera is not None

    def test_unconfigured_service_gives_clear_error(self):
        svc = Vision("bare")
        with pytest.raises(ValueError, match="no configuration applied"):
            svc._resolve_params(None)

    def test_reconfigure_drops_models_on_device_change(self, service):
        svc, camera = service
        svc._get_pipeline()
        assert svc._pipeline is not None
        attrs = dict(VALID_ATTRS, device="cpu")
        svc.reconfigure(make_config(attrs), {Camera.get_resource_name("cam"): camera})
        assert svc._pipeline is None


class TestRPCs:
    @pytest.mark.asyncio
    async def test_get_properties(self, service):
        svc, _ = service
        props = await svc.get_properties()
        assert props.detections_supported
        assert props.object_point_clouds_supported
        assert not props.classifications_supported

    @pytest.mark.asyncio
    async def test_get_detections_from_camera(self, service):
        svc, _ = service
        dets = await svc.get_detections_from_camera("cam")
        assert len(dets) == 1
        assert dets[0].class_name == "bolt"
        assert (dets[0].x_min, dets[0].y_min) == (20, 10)

    @pytest.mark.asyncio
    async def test_extra_queries_override(self, service):
        svc, _ = service
        dets = await svc.get_detections_from_camera("cam", extra={"queries": ["red gasket"]})
        assert dets[0].class_name == "red gasket"

    @pytest.mark.asyncio
    async def test_wrong_camera_name_rejected(self, service):
        svc, _ = service
        with pytest.raises(ValueError, match="configured for camera 'cam'"):
            await svc.get_detections_from_camera("other-cam")

    @pytest.mark.asyncio
    async def test_get_object_point_clouds(self, service):
        svc, _ = service
        objects = await svc.get_object_point_clouds("cam")
        assert len(objects) == 1
        obj = objects[0]
        assert obj.geometries.reference_frame == "cam"
        geo = obj.geometries.geometries[0]
        assert geo.label == "bolt"
        assert abs(geo.center.z - 1000) < 1  # mm
        pts = gs_pointcloud.decode_pcd(obj.point_cloud)
        assert pts.shape == (20 * 20, 3)
        np.testing.assert_allclose(pts[:, 2], 1.0, atol=1e-6)  # meters

    @pytest.mark.asyncio
    async def test_point_clouds_without_depth_error_names_camera(self, service, monkeypatch):
        svc, camera = service
        camera.with_depth = False
        with pytest.raises(ValueError, match="'cam' returned no depth image"):
            await svc.get_object_point_clouds("cam")

    @pytest.mark.asyncio
    async def test_point_clouds_without_intrinsics_error(self, service):
        svc, camera = service
        camera.with_intrinsics = False
        with pytest.raises(ValueError, match="intrinsic"):
            await svc.get_object_point_clouds("cam")

    @pytest.mark.asyncio
    async def test_capture_all(self, service):
        svc, _ = service
        result = await svc.capture_all_from_camera(
            "cam", return_image=True, return_detections=True,
            return_object_point_clouds=True,
        )
        assert result.image is not None
        assert len(result.detections) == 1
        assert len(result.objects) == 1

    @pytest.mark.asyncio
    async def test_do_command_segment_returns_masks(self, service):
        svc, _ = service
        resp = await svc.do_command({"segment": {"queries": ["bracket"], "return_masks": True}})
        objects = resp["objects"]
        assert len(objects) == 1
        obj = objects[0]
        assert obj["query"] == "bracket"
        assert obj["box_xyxy"] == [20.0, 10.0, 40.0, 30.0]
        mask_img = Image.open(io.BytesIO(base64.b64decode(obj["mask"])))
        assert mask_img.mode == "L" and mask_img.size == (W, H)
        mask_arr = np.array(mask_img)
        assert set(np.unique(mask_arr)) == {0, 255}

    @pytest.mark.asyncio
    async def test_do_command_unknown_command(self, service):
        svc, _ = service
        with pytest.raises(ValueError, match="unknown command"):
            await svc.do_command({"bogus": {}})
