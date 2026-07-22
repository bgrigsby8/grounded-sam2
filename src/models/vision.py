import asyncio
import base64
import io
from dataclasses import dataclass
from typing import Any, ClassVar, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from typing_extensions import Self
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.utils.pil import pil_to_viam_image, viam_to_pil_image
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    PointCloudObject,
    Pose,
    RectangularPrism,
    ResourceName,
    Vector3,
)
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import CaptureAllResult, Vision as VisionService
from viam.utils import ValueTypes, struct_to_dict

import pipeline as gs_pipeline
import pointcloud as gs_pointcloud

LOGGER = getLogger(__name__)

# Clouds larger than this are uniformly downsampled before PCD encoding.
MAX_CLOUD_POINTS = 50_000


@dataclass
class ServiceConfig:
    camera_name: str
    default_queries: List[str]
    box_threshold: float = gs_pipeline.DEFAULT_BOX_THRESHOLD
    text_threshold: float = gs_pipeline.DEFAULT_TEXT_THRESHOLD
    grounding_model: str = gs_pipeline.DEFAULT_GROUNDING_MODEL
    sam_model: str = gs_pipeline.DEFAULT_SAM_MODEL
    device: str = "auto"
    max_detections: int = gs_pipeline.DEFAULT_MAX_DETECTIONS
    min_mask_area_px: int = gs_pipeline.DEFAULT_MIN_MASK_AREA_PX

    @classmethod
    def from_proto(cls, config: ComponentConfig) -> Self:
        attrs = struct_to_dict(config.attributes)

        camera_name = attrs.get("camera_name")
        if not isinstance(camera_name, str) or not camera_name.strip():
            raise ValueError("`camera_name` is required and must be a non-empty string")

        queries = attrs.get("default_queries")
        if (
            not isinstance(queries, list)
            or not queries
            or not all(isinstance(q, str) and q.strip() for q in queries)
        ):
            raise ValueError(
                "`default_queries` is required and must be a non-empty list of non-empty strings"
            )

        def _float(key: str, default: float) -> float:
            val = attrs.get(key, default)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValueError(f"`{key}` must be a number, got {val!r}")
            return float(val)

        def _int(key: str, default: int) -> int:
            val = _float(key, default)
            if val != int(val) or val < 0:
                raise ValueError(f"`{key}` must be a non-negative integer")
            return int(val)

        def _str(key: str, default: str) -> str:
            val = attrs.get(key, default)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"`{key}` must be a non-empty string")
            return val

        return cls(
            camera_name=camera_name.strip(),
            default_queries=[q.strip() for q in queries],
            box_threshold=_float("box_threshold", gs_pipeline.DEFAULT_BOX_THRESHOLD),
            text_threshold=_float("text_threshold", gs_pipeline.DEFAULT_TEXT_THRESHOLD),
            grounding_model=_str("grounding_model", gs_pipeline.DEFAULT_GROUNDING_MODEL),
            sam_model=_str("sam_model", gs_pipeline.DEFAULT_SAM_MODEL),
            device=_str("device", "auto"),
            max_detections=_int("max_detections", gs_pipeline.DEFAULT_MAX_DETECTIONS),
            min_mask_area_px=_int("min_mask_area_px", gs_pipeline.DEFAULT_MIN_MASK_AREA_PX),
        )


class Vision(VisionService, EasyResource):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("brad-grigsby", "grounded-sam2"), "vision"
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._config: Optional[ServiceConfig] = None
        self._camera: Optional[Camera] = None
        self._pipeline: Optional[gs_pipeline.GroundedSam2Pipeline] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        return super().new(config, dependencies)

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        cfg = ServiceConfig.from_proto(config)
        # The camera is an implicit required dependency so viam-server wires it up.
        return [cfg.camera_name], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        cfg = ServiceConfig.from_proto(config)

        camera = dependencies.get(Camera.get_resource_name(cfg.camera_name))
        if camera is None or not isinstance(camera, Camera):
            raise ValueError(
                f"camera '{cfg.camera_name}' not found in dependencies; "
                "make sure it exists on the machine"
            )
        self._camera = camera

        # Model/device changes invalidate the cached weights; they reload
        # lazily on the next inference with the new settings.
        old = self._config
        if old is not None and self._pipeline is not None:
            if (
                old.grounding_model != cfg.grounding_model
                or old.sam_model != cfg.sam_model
                or old.device != cfg.device
            ):
                LOGGER.info("model/device config changed; dropping cached models")
                self._pipeline.unload()
                self._pipeline = None

        self._config = cfg

    # ------------------------------------------------------------------
    # helpers

    def _get_pipeline(self) -> gs_pipeline.GroundedSam2Pipeline:
        assert self._config is not None
        if self._pipeline is None:
            cfg = self._config
            self._pipeline = gs_pipeline.GroundedSam2Pipeline(
                grounding_model=cfg.grounding_model,
                sam_model=cfg.sam_model,
                device=cfg.device,
                box_threshold=cfg.box_threshold,
                text_threshold=cfg.text_threshold,
                max_detections=cfg.max_detections,
                min_mask_area_px=cfg.min_mask_area_px,
            )
        return self._pipeline

    def _resolve_params(self, extra: Optional[Mapping[str, Any]]) -> dict:
        """Merge config defaults with per-request `extra` overrides."""
        assert self._config is not None
        cfg = self._config
        params = {
            "queries": list(cfg.default_queries),
            "box_threshold": cfg.box_threshold,
            "text_threshold": cfg.text_threshold,
            "max_detections": cfg.max_detections,
            "min_mask_area_px": cfg.min_mask_area_px,
        }
        if not extra:
            return params

        if "queries" in extra:
            queries = extra["queries"]
            if (
                not isinstance(queries, (list, tuple))
                or not queries
                or not all(isinstance(q, str) and q.strip() for q in queries)
            ):
                raise ValueError(
                    "extra['queries'] must be a non-empty list of non-empty strings"
                )
            params["queries"] = [q.strip() for q in queries]

        for key in ("box_threshold", "text_threshold"):
            if key in extra:
                val = extra[key]
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise ValueError(f"extra['{key}'] must be a number")
                params[key] = float(val)

        for key in ("max_detections", "min_mask_area_px"):
            if key in extra:
                val = extra[key]
                if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
                    raise ValueError(f"extra['{key}'] must be a non-negative number")
                params[key] = int(val)

        return params

    def _require_camera(self, camera_name: str) -> Camera:
        assert self._config is not None
        if self._camera is None:
            raise ValueError("service is not configured with a camera")
        if camera_name and camera_name != self._config.camera_name:
            raise ValueError(
                f"this service is configured for camera '{self._config.camera_name}', "
                f"but was asked about camera '{camera_name}'"
            )
        return self._camera

    async def _get_rgb_and_depth(
        self, camera: Camera
    ) -> Tuple[Image.Image, Optional[np.ndarray]]:
        """Fetch a time-consistent color+depth pair via one get_images call."""
        assert self._config is not None
        images, _meta = await camera.get_images()

        rgb: Optional[Image.Image] = None
        depth: Optional[np.ndarray] = None
        for named in images:
            if named.mime_type == CameraMimeType.VIAM_RAW_DEPTH:
                if depth is None:
                    depth = gs_pointcloud.decode_viam_raw_depth(named.data)
            elif rgb is None and named.mime_type in (
                CameraMimeType.JPEG,
                CameraMimeType.PNG,
                CameraMimeType.VIAM_RGBA,
            ):
                rgb = viam_to_pil_image(
                    ViamImage(named.data, named.mime_type)
                ).convert("RGB")

        if rgb is None:
            raise ValueError(
                f"camera '{self._config.camera_name}' returned no color image "
                "from get_images()"
            )
        return rgb, depth

    async def _get_intrinsics(self, camera: Camera) -> Tuple[float, float, float, float]:
        assert self._config is not None
        props = await camera.get_properties()
        ip = props.intrinsic_parameters
        if ip is None or ip.focal_x_px == 0 or ip.focal_y_px == 0:
            raise ValueError(
                f"camera '{self._config.camera_name}' does not report intrinsic "
                "parameters (focal lengths); they are required to back-project "
                "depth to 3D points"
            )
        return ip.focal_x_px, ip.focal_y_px, ip.center_x_px, ip.center_y_px

    @staticmethod
    def _to_detection_proto(det: gs_pipeline.Detection2D, width: int, height: int) -> Detection:
        x_min, y_min, x_max, y_max = det.box_xyxy
        return Detection(
            x_min=int(round(x_min)),
            y_min=int(round(y_min)),
            x_max=int(round(x_max)),
            y_max=int(round(y_max)),
            x_min_normalized=x_min / width,
            y_min_normalized=y_min / height,
            x_max_normalized=x_max / width,
            y_max_normalized=y_max / height,
            confidence=det.score,
            class_name=det.query,
        )

    def _segments_to_point_clouds(
        self,
        segments: List[gs_pipeline.Segment2D],
        depth_mm: np.ndarray,
        intrinsics: Tuple[float, float, float, float],
        reference_frame: str,
    ) -> List[PointCloudObject]:
        """Mask depth per segment, back-project, and package PCD + bounding box.

        Point cloud bytes are PCD in meters (PCL convention); geometry center
        and dims are millimeters (Viam convention). Both in the camera frame.
        """
        fx, fy, cx, cy = intrinsics
        objects: List[PointCloudObject] = []
        for seg in segments:
            mask = seg.mask
            if mask.shape != depth_mm.shape:
                # Depth stream at a different resolution than color: rescale
                # the mask; intrinsics must correspond to the depth image.
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_img = mask_img.resize(
                    (depth_mm.shape[1], depth_mm.shape[0]), Image.NEAREST
                )
                mask = np.array(mask_img) > 0

            points_mm = gs_pointcloud.depth_to_points(depth_mm, fx, fy, cx, cy, mask=mask)
            if points_mm.shape[0] == 0:
                LOGGER.warning(
                    "detection '%s' has no valid depth readings; skipping", seg.query
                )
                continue
            points_mm = gs_pointcloud.downsample_points(points_mm, MAX_CLOUD_POINTS)

            pcd_bytes = gs_pointcloud.encode_pcd(points_mm, scale=0.001)
            center, dims = gs_pointcloud.axis_aligned_box(points_mm)
            geometry = Geometry(
                center=Pose(
                    x=float(center[0]),
                    y=float(center[1]),
                    z=float(center[2]),
                    o_z=1.0,
                ),
                box=RectangularPrism(
                    dims_mm=Vector3(
                        x=float(dims[0]), y=float(dims[1]), z=float(dims[2])
                    )
                ),
                label=seg.query,
            )
            objects.append(
                PointCloudObject(
                    point_cloud=pcd_bytes,
                    geometries=GeometriesInFrame(
                        reference_frame=reference_frame, geometries=[geometry]
                    ),
                )
            )
        return objects

    # ------------------------------------------------------------------
    # vision API

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> VisionService.Properties:
        return VisionService.Properties(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=True,
        )

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        params = self._resolve_params(extra)
        rgb = viam_to_pil_image(image).convert("RGB")
        detections = await asyncio.to_thread(
            self._get_pipeline().detect,
            rgb,
            params["queries"],
            box_threshold=params["box_threshold"],
            text_threshold=params["text_threshold"],
            max_detections=params["max_detections"],
        )
        return [self._to_detection_proto(d, rgb.width, rgb.height) for d in detections]

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        camera = self._require_camera(camera_name)
        image = await camera.get_image(mime_type=CameraMimeType.JPEG)
        return await self.get_detections(image, extra=extra, timeout=timeout)

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        assert self._config is not None
        camera = self._require_camera(camera_name)
        params = self._resolve_params(extra)

        rgb, depth_mm = await self._get_rgb_and_depth(camera)
        if depth_mm is None:
            raise ValueError(
                f"camera '{self._config.camera_name}' returned no depth image: "
                "get_object_point_clouds requires an RGBD camera whose "
                "get_images() includes an image/vnd.viam.dep depth source"
            )
        intrinsics = await self._get_intrinsics(camera)

        segments = await asyncio.to_thread(
            self._get_pipeline().segment,
            rgb,
            params["queries"],
            box_threshold=params["box_threshold"],
            text_threshold=params["text_threshold"],
            max_detections=params["max_detections"],
            min_mask_area_px=params["min_mask_area_px"],
        )
        return await asyncio.to_thread(
            self._segments_to_point_clouds,
            segments,
            depth_mm,
            intrinsics,
            self._config.camera_name,
        )

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        assert self._config is not None
        camera = self._require_camera(camera_name)
        result = CaptureAllResult()

        if not (return_image or return_detections or return_object_point_clouds):
            return result

        params = self._resolve_params(extra)
        rgb, depth_mm = await self._get_rgb_and_depth(camera)

        if return_image:
            result.image = pil_to_viam_image(rgb, CameraMimeType.JPEG)

        if return_object_point_clouds:
            # One segmentation pass serves both detections and point clouds.
            if depth_mm is None:
                raise ValueError(
                    f"camera '{self._config.camera_name}' returned no depth image: "
                    "object point clouds require an RGBD camera whose "
                    "get_images() includes an image/vnd.viam.dep depth source"
                )
            intrinsics = await self._get_intrinsics(camera)
            segments = await asyncio.to_thread(
                self._get_pipeline().segment,
                rgb,
                params["queries"],
                box_threshold=params["box_threshold"],
                text_threshold=params["text_threshold"],
                max_detections=params["max_detections"],
                min_mask_area_px=params["min_mask_area_px"],
            )
            result.objects = await asyncio.to_thread(
                self._segments_to_point_clouds,
                segments,
                depth_mm,
                intrinsics,
                self._config.camera_name,
            )
            if return_detections:
                result.detections = [
                    self._to_detection_proto(
                        gs_pipeline.Detection2D(s.query, s.score, s.box_xyxy),
                        rgb.width,
                        rgb.height,
                    )
                    for s in segments
                ]
        elif return_detections:
            detections = await asyncio.to_thread(
                self._get_pipeline().detect,
                rgb,
                params["queries"],
                box_threshold=params["box_threshold"],
                text_threshold=params["text_threshold"],
                max_detections=params["max_detections"],
            )
            result.detections = [
                self._to_detection_proto(d, rgb.width, rgb.height) for d in detections
            ]

        return result

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError("classifications are not supported by this service")

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError("classifications are not supported by this service")

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        if "segment" in command:
            return await self._cmd_segment(command["segment"])
        raise ValueError(
            f"unknown command(s) {list(command.keys())!r}; supported: 'segment'"
        )

    async def _cmd_segment(self, args: ValueTypes) -> Mapping[str, ValueTypes]:
        """`{"segment": {"queries": [...], "return_masks": true}}`.

        Returns per-object query/score/box, plus the raw 2D mask as a
        base64-encoded single-channel PNG (0/255) when return_masks is set.
        This is the escape hatch for callers that want masks, which the
        standard vision API has no return type for.
        """
        if not isinstance(args, Mapping):
            raise ValueError("'segment' command value must be an object")
        return_masks = bool(args.get("return_masks", False))
        params = self._resolve_params(args)

        camera = self._require_camera(str(args.get("camera_name", "")))
        rgb, _depth = await self._get_rgb_and_depth(camera)

        segments = await asyncio.to_thread(
            self._get_pipeline().segment,
            rgb,
            params["queries"],
            box_threshold=params["box_threshold"],
            text_threshold=params["text_threshold"],
            max_detections=params["max_detections"],
            min_mask_area_px=params["min_mask_area_px"],
        )

        objects: List[Mapping[str, ValueTypes]] = []
        for seg in segments:
            obj: dict = {
                "query": seg.query,
                "score": seg.score,
                "box_xyxy": [float(v) for v in seg.box_xyxy],
            }
            if return_masks:
                png = io.BytesIO()
                Image.fromarray(seg.mask.astype(np.uint8) * 255, mode="L").save(
                    png, format="PNG"
                )
                obj["mask"] = base64.b64encode(png.getvalue()).decode("ascii")
            objects.append(obj)
        return {"objects": objects}
