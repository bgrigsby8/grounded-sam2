# Module grounded-sam2

Open-vocabulary segmentation as a Viam vision service. Tell your robot what to
pick in natural language ("black plastic bracket") and get back 2D detections,
per-object segmentation masks, and — with an RGBD camera — segmented 3D object
point clouds ready for Viam's motion planning.

Under the hood: [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
turns text queries into bounding boxes, and [SAM 2](https://huggingface.co/facebook/sam2.1-hiera-small)
turns those boxes into pixel-accurate masks. For 3D, each mask is applied to
the aligned depth image and back-projected through the camera intrinsics into
a per-object point cloud in the camera frame.

> **GPU strongly recommended.** On CPU expect several seconds per frame even
> with the default (smallest) models. CUDA and Apple Silicon (`mps`) are
> auto-detected. Model weights (~1 GB with the defaults) download from
> Hugging Face on first inference and are cached.

## Models

This module provides the following model(s):

- [`brad-grigsby:grounded-sam2:vision`](brad-grigsby_grounded-sam2_vision.md) — open-vocabulary detection, segmentation, and object point clouds

## Configuration

```json
{
  "camera_name": "rgbd-cam",
  "default_queries": ["black plastic bracket", "hex bolt"],
  "box_threshold": 0.35,
  "text_threshold": 0.25,
  "grounding_model": "IDEA-Research/grounding-dino-tiny",
  "sam_model": "facebook/sam2.1-hiera-small",
  "device": "auto",
  "max_detections": 10,
  "min_mask_area_px": 100
}
```

| Name | Type | Inclusion | Default | Description |
|------|------|-----------|---------|-------------|
| `camera_name` | string | **Required** | — | Camera resource this service reads from. Added as a dependency automatically. Must be an RGBD camera (color + `image/vnd.viam.dep` depth) for `GetObjectPointClouds`. |
| `default_queries` | list of strings | **Required** | — | Text prompts used when a request doesn't override them. |
| `box_threshold` | float | Optional | `0.35` | Minimum Grounding DINO box confidence. |
| `text_threshold` | float | Optional | `0.25` | Minimum Grounding DINO text-match confidence. |
| `grounding_model` | string | Optional | `IDEA-Research/grounding-dino-tiny` | Hugging Face id; use `IDEA-Research/grounding-dino-base` for accuracy over speed. |
| `sam_model` | string | Optional | `facebook/sam2.1-hiera-small` | Hugging Face id; use `facebook/sam2.1-hiera-large` for accuracy over speed. |
| `device` | string | Optional | `auto` | `auto` (cuda → mps → cpu), or an explicit torch device like `cuda`, `cuda:1`, `mps`, `cpu`. |
| `max_detections` | int | Optional | `10` | Keep at most this many detections (highest score first). |
| `min_mask_area_px` | int | Optional | `100` | Drop masks smaller than this many pixels. |

### Per-request overrides via `extra`

Every vision API call accepts an `extra` dict that can override the config for
that one request — this is the main way to do dynamic targeting:

```python
detections = await vision.get_detections_from_camera(
    "rgbd-cam",
    extra={"queries": ["red gasket"], "box_threshold": 0.4},
)
```

Supported keys: `queries`, `box_threshold`, `text_threshold`,
`max_detections`, `min_mask_area_px`.

## API

Implements [`rdk:service:vision`](https://docs.viam.com/dev/reference/apis/services/vision/):

- **`GetDetections` / `GetDetectionsFromCamera`** — Grounding DINO only (no
  SAM, fast path). `class_name` is the matched query, `confidence` is the
  DINO score.
- **`GetObjectPointClouds`** — full pipeline. Returns one `PointCloudObject`
  per detected object: PCD-encoded points (meters, camera frame, downsampled
  to ≤50k points) plus an axis-aligned bounding box (millimeters, camera
  frame) labeled with the query. Requires depth + intrinsics; raises a clear
  error naming the camera if either is missing.
- **`CaptureAllFromCamera`** — image, detections, and object point clouds in
  one inference pass, per the request flags.
- **`GetProperties`** — detections ✓, object point clouds ✓, classifications ✗.
- **`DoCommand`** — raw-mask escape hatch, see below.

### DoCommand: `segment`

The standard vision API has no mask return type; this returns raw 2D masks
for your own grasp-point logic:

```json
{
  "segment": {
    "queries": ["bracket"],
    "return_masks": true
  }
}
```

Optional keys: `camera_name`, `box_threshold`, `text_threshold`,
`max_detections`, `min_mask_area_px` (same semantics as `extra`).

Response, per object:

```json
{
  "objects": [
    {
      "query": "bracket",
      "score": 0.62,
      "box_xyxy": [211.0, 140.5, 388.2, 305.9],
      "mask": "<base64 PNG, single channel, 0/255>"
    }
  ]
}
```

## Development

```sh
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt pytest pytest-asyncio
./venv/bin/python -m pytest tests/                       # full suite (downloads ~1 GB of weights)
GS2_SKIP_MODEL_TESTS=1 ./venv/bin/python -m pytest tests/  # fast tests only
```

`src/pipeline.py` (inference) and `src/pointcloud.py` (back-projection/PCD)
have no Viam SDK imports and can be exercised standalone; the model test
writes `tests/output/overlay.png` for eyeballing masks.
