"""Visualize grounded-sam2 segmentation against a live machine.

Fetches a frame from the camera, runs the vision service's `segment`
DoCommand, and saves (and opens) an overlay PNG with each object's mask
tinted, its box drawn, and its query + score labeled.

Usage:
    export VIAM_ADDRESS="my-machine-main.abc123.viam.cloud"
    export VIAM_API_KEY="..."
    export VIAM_API_KEY_ID="..."
    ./venv/bin/python scripts/visualize.py \
        --vision vision-1 --camera rgbd-cam \
        --queries "phone" "coffee mug" \
        --box-threshold 0.5 \
        -o overlay.png
"""

import argparse
import asyncio
import base64
import io
import os
import platform
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from viam.components.camera import Camera
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import CameraMimeType, ViamImage
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient

COLORS = [
    (255, 60, 60),
    (60, 220, 60),
    (80, 120, 255),
    (255, 200, 40),
    (220, 60, 220),
    (40, 220, 220),
]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", default="vision-1", help="vision service name")
    parser.add_argument("--camera", default="rgbd-cam", help="camera name")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="text queries (default: the service's default_queries)")
    parser.add_argument("--box-threshold", type=float, default=None)
    parser.add_argument("-o", "--output", default="overlay.png")
    parser.add_argument("--no-open", action="store_true", help="don't open the result")
    args = parser.parse_args()

    address = os.environ.get("VIAM_ADDRESS")
    api_key = os.environ.get("VIAM_API_KEY")
    api_key_id = os.environ.get("VIAM_API_KEY_ID")
    if not (address and api_key and api_key_id):
        print("set VIAM_ADDRESS, VIAM_API_KEY, and VIAM_API_KEY_ID", file=sys.stderr)
        return 1

    machine = await RobotClient.at_address(
        address, RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
    )
    try:
        camera = Camera.from_robot(machine, args.camera)
        vision = VisionClient.from_robot(machine, args.vision)

        images, _meta = await camera.get_images()
        named = next(
            (im for im in images if im.mime_type in
             (CameraMimeType.JPEG, CameraMimeType.PNG, CameraMimeType.VIAM_RGBA)),
            None,
        )
        if named is None:
            print(f"camera {args.camera!r} returned no color image", file=sys.stderr)
            return 1
        image = viam_to_pil_image(ViamImage(named.data, named.mime_type)).convert("RGB")

        segment_args: dict = {"return_masks": True}
        if args.queries:
            segment_args["queries"] = args.queries
        if args.box_threshold is not None:
            segment_args["box_threshold"] = args.box_threshold
        response = await vision.do_command({"segment": segment_args})
    finally:
        await machine.close()

    objects = response.get("objects", [])
    print(f"{len(objects)} object(s)")

    overlay = np.array(image, dtype=np.float32)
    for i, obj in enumerate(objects):
        color = COLORS[i % len(COLORS)]
        mask = np.array(
            Image.open(io.BytesIO(base64.b64decode(obj["mask"]))).resize(image.size)
        ) > 0
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array(color, dtype=np.float32)
        pct = 100.0 * mask.mean()
        print(f"  [{i}] {obj['query']!r} score={obj['score']:.2f} "
              f"box={[round(v, 1) for v in obj['box_xyxy']]} mask={pct:.1f}% of frame")

    out = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for i, obj in enumerate(objects):
        color = COLORS[i % len(COLORS)]
        x0, y0, x1, y1 = obj["box_xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{obj['query']} {obj['score']:.2f}"
        text_box = draw.textbbox((x0 + 3, max(0, y0 - 16)), label)
        draw.rectangle(text_box, fill=color)
        draw.text((x0 + 3, max(0, y0 - 16)), label, fill=(0, 0, 0))

    out.save(args.output)
    print(f"saved {args.output}")
    if not args.no_open and platform.system() == "Darwin":
        subprocess.run(["open", args.output], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
