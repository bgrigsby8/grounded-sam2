"""Grounded SAM 2 inference pipeline.

Grounding DINO (open-vocabulary detection) -> SAM 2 (per-box segmentation).

This module has no Viam SDK imports so it can be tested standalone against a
sample image before ever touching a robot. Heavy dependencies (torch,
transformers) are imported lazily so that importing this module — and starting
the Viam module process — stays fast; models load on first inference and are
cached until `unload()`.
"""

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)

DEFAULT_GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SAM_MODEL = "facebook/sam2.1-hiera-small"
DEFAULT_BOX_THRESHOLD = 0.35
DEFAULT_TEXT_THRESHOLD = 0.25
DEFAULT_MAX_DETECTIONS = 10
DEFAULT_MIN_MASK_AREA_PX = 100


@dataclass(frozen=True)
class Detection2D:
    """One Grounding DINO detection. Box is pixel-space (x_min, y_min, x_max, y_max)."""

    query: str
    score: float
    box_xyxy: Tuple[float, float, float, float]


@dataclass(frozen=True)
class Segment2D:
    """One detection plus its SAM 2 mask (bool array, HxW, image-sized)."""

    query: str
    score: float
    box_xyxy: Tuple[float, float, float, float]
    mask: np.ndarray


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_prompt(queries: Sequence[str]) -> str:
    """Grounding DINO expects lowercase, period-separated phrases in one string."""
    cleaned = [q.strip().lower().rstrip(".") for q in queries if q.strip()]
    if not cleaned:
        raise ValueError("queries must contain at least one non-empty string")
    return ". ".join(cleaned) + "."


class _TransformersSam2Backend:
    """SAM 2 via native `transformers` support (Sam2Model/Sam2Processor)."""

    def __init__(self, model_id: str, device: str):
        from transformers import Sam2Model, Sam2Processor

        self._processor = Sam2Processor.from_pretrained(model_id)
        self._model = Sam2Model.from_pretrained(model_id).to(device).eval()
        self._device = device

    def masks_for_boxes(self, image: Image.Image, boxes_xyxy: Sequence[Sequence[float]]) -> np.ndarray:
        import torch

        inputs = self._processor(
            images=image,
            input_boxes=[[list(map(float, b)) for b in boxes_xyxy]],
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=False)
        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
        )[0]
        # (num_boxes, num_masks_per_box, H, W) -> best mask per box
        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[:, 0]
        return masks.astype(bool)


class _OfficialSam2Backend:
    """SAM 2 via the facebookresearch `sam2` package (fallback)."""

    def __init__(self, model_id: str, device: str):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)

    def masks_for_boxes(self, image: Image.Image, boxes_xyxy: Sequence[Sequence[float]]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            self._predictor.set_image(np.array(image.convert("RGB")))
            masks, _, _ = self._predictor.predict(
                box=np.array(boxes_xyxy, dtype=np.float32),
                multimask_output=False,
            )
        masks = np.asarray(masks)
        if masks.ndim == 4:  # (num_boxes, 1, H, W)
            masks = masks[:, 0]
        elif masks.ndim == 2:  # single box -> (H, W)
            masks = masks[None]
        return masks.astype(bool)


def _load_sam_backend(model_id: str, device: str):
    try:
        return _TransformersSam2Backend(model_id, device)
    except ImportError:
        LOGGER.info(
            "transformers has no native SAM 2 support; falling back to the `sam2` package"
        )
    return _OfficialSam2Backend(model_id, device)


class GroundedSam2Pipeline:
    """Lazy-loading Grounding DINO + SAM 2 pipeline.

    Thread-safe: model loading and inference are serialized with a lock, since
    a single set of model weights must not run concurrent forward passes.
    """

    def __init__(
        self,
        grounding_model: str = DEFAULT_GROUNDING_MODEL,
        sam_model: str = DEFAULT_SAM_MODEL,
        device: str = "auto",
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        min_mask_area_px: int = DEFAULT_MIN_MASK_AREA_PX,
    ):
        self.grounding_model = grounding_model
        self.sam_model = sam_model
        self.device_setting = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_detections = max_detections
        self.min_mask_area_px = min_mask_area_px

        self._lock = threading.Lock()
        self._device: Optional[str] = None
        self._dino_processor = None
        self._dino_model = None
        self._sam_backend = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self.device_setting)
        return self._device

    def _ensure_dino(self) -> None:
        if self._dino_model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        LOGGER.info("loading %s on %s", self.grounding_model, self.device)
        self._dino_processor = AutoProcessor.from_pretrained(self.grounding_model)
        self._dino_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_model)
            .to(self.device)
            .eval()
        )

    def _ensure_sam(self) -> None:
        if self._sam_backend is not None:
            return
        LOGGER.info("loading %s on %s", self.sam_model, self.device)
        self._sam_backend = _load_sam_backend(self.sam_model, self.device)

    def unload(self) -> None:
        """Drop cached models (e.g. after a reconfigure changed model/device)."""
        with self._lock:
            self._dino_processor = None
            self._dino_model = None
            self._sam_backend = None
            self._device = None

    def detect(
        self,
        image: Image.Image,
        queries: Sequence[str],
        *,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> List[Detection2D]:
        """Run Grounding DINO only (no SAM). Returns detections sorted by score."""
        with self._lock:
            return self._detect_locked(
                image.convert("RGB"),
                queries,
                box_threshold if box_threshold is not None else self.box_threshold,
                text_threshold if text_threshold is not None else self.text_threshold,
                max_detections if max_detections is not None else self.max_detections,
            )

    def segment(
        self,
        image: Image.Image,
        queries: Sequence[str],
        *,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        min_mask_area_px: Optional[int] = None,
    ) -> List[Segment2D]:
        """Full pipeline: DINO boxes -> SAM 2 masks. Drops masks below min area."""
        min_area = (
            min_mask_area_px if min_mask_area_px is not None else self.min_mask_area_px
        )
        with self._lock:
            rgb = image.convert("RGB")
            detections = self._detect_locked(
                rgb,
                queries,
                box_threshold if box_threshold is not None else self.box_threshold,
                text_threshold if text_threshold is not None else self.text_threshold,
                max_detections if max_detections is not None else self.max_detections,
            )
            if not detections:
                return []
            self._ensure_sam()
            masks = self._sam_backend.masks_for_boxes(
                rgb, [d.box_xyxy for d in detections]
            )
            segments = []
            for det, mask in zip(detections, masks):
                if int(mask.sum()) < min_area:
                    continue
                segments.append(
                    Segment2D(
                        query=det.query,
                        score=det.score,
                        box_xyxy=det.box_xyxy,
                        mask=mask,
                    )
                )
            return segments

    def _detect_locked(
        self,
        rgb: Image.Image,
        queries: Sequence[str],
        box_threshold: float,
        text_threshold: float,
        max_detections: int,
    ) -> List[Detection2D]:
        import torch

        self._ensure_dino()
        prompt = build_prompt(queries)
        # Map the lowercase phrase DINO echoes back to the user's original query.
        original_by_lower = {q.strip().lower().rstrip("."): q for q in queries if q.strip()}

        inputs = self._dino_processor(images=rgb, text=prompt, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self._dino_model(**inputs)
        results = self._dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[rgb.size[::-1]],
        )[0]

        labels = results.get("text_labels", results.get("labels"))
        detections = []
        for score, label, box in zip(results["scores"], labels, results["boxes"]):
            label = str(label).strip()
            if not label:
                continue
            detections.append(
                Detection2D(
                    query=original_by_lower.get(label, label),
                    score=float(score),
                    box_xyxy=tuple(float(v) for v in box.tolist()),
                )
            )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[:max_detections]
