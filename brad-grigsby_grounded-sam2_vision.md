# Model brad-grigsby:grounded-sam2:vision

Open-vocabulary vision service: Grounding DINO (text → boxes) + SAM 2
(boxes → masks) + depth back-projection (masks → per-object 3D point clouds).
Query objects in natural language at request time; feed the returned
`PointCloudObject`s straight into grasp/motion logic.

## Configuration

```json
{
  "camera_name": "<string>",
  "default_queries": ["<string>", "..."],
  "box_threshold": <float>,
  "text_threshold": <float>,
  "grounding_model": "<string>",
  "sam_model": "<string>",
  "device": "<string>",
  "max_detections": <int>,
  "min_mask_area_px": <int>
}
```

### Attributes

| Name | Type | Inclusion | Default | Description |
|------|------|-----------|---------|-------------|
| `camera_name` | string | Required | — | Camera resource to read from (RGBD needed for point clouds). |
| `default_queries` | list of strings | Required | — | Text prompts used when a request doesn't override them. |
| `box_threshold` | float | Optional | `0.35` | Minimum box confidence. |
| `text_threshold` | float | Optional | `0.25` | Minimum text-match confidence. |
| `grounding_model` | string | Optional | `IDEA-Research/grounding-dino-tiny` | Grounding DINO Hugging Face id. |
| `sam_model` | string | Optional | `facebook/sam2.1-hiera-small` | SAM 2 Hugging Face id. |
| `device` | string | Optional | `auto` | `auto`, `cuda[:N]`, `mps`, or `cpu`. |
| `max_detections` | int | Optional | `10` | Cap on detections per request. |
| `min_mask_area_px` | int | Optional | `100` | Drop masks smaller than this. |

### Example Configuration

```json
{
  "camera_name": "rgbd-cam",
  "default_queries": ["black plastic bracket", "hex bolt"]
}
```

Every vision API call also accepts `extra` overrides, e.g.
`extra={"queries": ["red gasket"], "box_threshold": 0.4}`.

## DoCommand

### `segment`

Returns raw 2D masks (the standard vision API has no mask type):

```json
{
  "segment": {
    "queries": ["bracket"],
    "return_masks": true
  }
}
```

Response: `{"objects": [{"query", "score", "box_xyxy", "mask"}]}` where
`mask` is a base64-encoded single-channel PNG (0/255).
