# Viam Module: Grounded SAM 2 Open-Vocabulary Segmentation

## Goal

Build a Viam **vision service** module that wraps the Grounded SAM 2 pipeline (Grounding DINO for open-vocabulary detection → SAM 2 for mask generation) so that a robot can be told, in natural language, what to pick ("black plastic bracket") and get back 2D detections, per-object segmentation masks, and — when paired with an RGBD camera — segmented 3D object point clouds that can be fed directly into Viam's motion planning for pick and place.

- **Repo name:** `grounded-sam2`
- **Model triple:** `viam:grounded-sam2:vision`
- **Language:** Python (Viam Python SDK), since Grounding DINO and SAM 2 are PyTorch models
- **API implemented:** `rdk:service:vision`

## Why a vision service (not a camera)

The Viam vision service API already has the right surface for this: `GetDetections` for boxes, `GetObjectPointClouds` for segmented 3D objects, and `DoCommand` for anything nonstandard (raw masks, dynamic prompts). `GetObjectPointClouds` is the important one for pick and place — its output (per-object PCD + bounding geometry) is what downstream grasp logic and the motion service consume.

## Pipeline (per inference)

1. Grab an RGB frame (and depth, when needed) from a configured camera resource.
2. Run **Grounding DINO** with a text prompt → bounding boxes + confidence scores.
3. Prompt **SAM 2** with those boxes → one binary mask per detection.
4. For 3D: apply each mask to the aligned depth image, back-project masked pixels to 3D using the camera intrinsics, and emit one point cloud per object.

## Model choices

Prefer Hugging Face `transformers` implementations where available so we avoid git-installed research repos:

- Grounding DINO: `IDEA-Research/grounding-dino-tiny` by default (configurable to `grounding-dino-base` for accuracy). Available via `AutoModelForZeroShotObjectDetection` in transformers.
- SAM 2: `facebook/sam2.1-hiera-small` by default (configurable to `hiera-large`). Check whether the installed `transformers` version has native SAM 2 support; if not, fall back to the official `sam2` package from facebookresearch. Isolate this behind a small adapter class so the rest of the module doesn't care which backend loaded.

Models must **lazy-load on first inference** (not at module startup) and be cached. Device selection: `auto` (cuda if available → mps → cpu), overridable in config.

## Configuration attributes

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

- `camera_name` (required): camera resource this service reads from; must be declared as a dependency in `validate_config` so viam-server wires it up.
- `default_queries` (required, non-empty): the text prompts used when a request doesn't override them.
- All others optional with the defaults shown.
- **Per-request override:** every vision API call accepts an `extra` dict. Support `extra = {"queries": ["red gasket"], "box_threshold": 0.4}` to override config at request time. This is the main way users will do dynamic targeting, so make sure it's plumbed through all methods and documented in the README.

## API mapping

### `get_properties`
Return `detections_supported=True`, `classifications_supported=False`, `object_point_clouds_supported=True`.

### `get_detections` / `get_detections_from_camera`
Run steps 1–2 (Grounding DINO only — skip SAM for speed). Return `Detection` protos with the matched text query as `class_name` and the DINO score as `confidence`.

### `get_object_point_clouds`
Full pipeline including depth back-projection. For each detection:
- Fetch color + depth in one call (`camera.get_images()`) so the pair is time-consistent; get intrinsics from the camera's properties.
- Mask the depth image, back-project to 3D in the camera frame, downsample if the cloud exceeds ~50k points.
- Return `PointCloudObject`s: PCD-encoded bytes plus an axis-aligned bounding box geometry with the query string as the label.
- If the camera provides no depth, raise a clear, actionable error naming the camera and what it lacks.

### `capture_all_from_camera`
Support returning image + detections + object point clouds per the request flags, reusing one inference pass.

### `do_command`
Implement one command:

```json
{"segment": {"queries": ["bracket"], "return_masks": true}}
```

Response: per-object `{query, score, box_xyxy, mask}` where `mask` is a base64-encoded PNG (single channel, 0/255). This is the escape hatch for users who want raw 2D masks (e.g., for their own grasp-point logic) since the standard vision API has no mask return type.

## Repo layout

```
grounded-sam2/
├── meta.json
├── pyproject.toml or requirements.txt
├── build.sh                  # viam module build entrypoint
├── run.sh                    # module entrypoint (venv bootstrap + exec)
├── src/
│   ├── main.py               # module registration + serve
│   ├── grounded_sam2.py      # Vision subclass: config, validation, API methods
│   ├── pipeline.py           # model loading adapter + inference (no Viam imports)
│   └── pointcloud.py         # depth back-projection + PCD encoding
├── tests/
│   ├── test_pipeline.py      # runs on a fixture image, no robot needed
│   └── test_pointcloud.py    # synthetic depth → known 3D points
└── README.md
```

Keep `pipeline.py` free of Viam SDK imports so it can be tested standalone with a sample image before ever touching a robot.

## Dependencies

`viam-sdk`, `torch`, `torchvision`, `transformers`, `numpy`, `pillow`. Avoid `open3d` — write the PCD encoder by hand (it's ~30 lines for ascii/binary XYZ) to keep the install light. Pin versions in requirements.

## Implementation notes / gotchas

- Grounding DINO prompt format: queries must be lowercase and period-separated when batched into one prompt string ("black plastic bracket. hex bolt."). Handle this inside `pipeline.py`; users pass a plain list.
- Run inference in a thread executor (`asyncio.to_thread`) — the Viam SDK service methods are async and a multi-second forward pass must not block the event loop.
- Reconfigure must handle model-name/device changes by dropping the cached models so they reload lazily with new settings.
- `validate_config` should return `camera_name` as an implicit dependency and reject empty `default_queries`.
- meta.json: `visibility: private` to start, models list with `viam:grounded-sam2:vision`, entrypoint `run.sh`. Follow current Viam module packaging conventions — check the latest Viam docs for the Python module build/upload flow rather than assuming.
- README must include: a sample service config JSON, an example `extra` override, the `do_command` schema, and a note that GPU is strongly recommended (expect several seconds per frame on CPU with the small models).

## Milestones

1. `pipeline.py` working standalone on a fixture image (detections + masks, saved as overlay PNG for eyeballing).
2. Vision service wrapper with `get_detections` + `do_command` masks, tested against a live camera via local module reload.
3. `get_object_point_clouds` with depth back-projection; verify returned clouds land in sensible positions relative to the camera frame.
4. Packaging (meta.json, build/run scripts), README, and registry-ready cleanup.

## Out of scope (for now)

Grasp pose generation (GraspGen/AnyGrasp integration is a likely follow-up module), tracking across frames (SAM 2 video mode), and any fine-tuning. Design `pipeline.py` so a grasp-pose stage could consume its masks + clouds later.