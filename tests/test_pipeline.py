"""Pipeline tests.

The fast tests always run. The model tests download Grounding DINO +
SAM 2 weights from Hugging Face on first run (~1 GB) and run a real
inference on a fixture image; skip them with GS2_SKIP_MODEL_TESTS=1.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "cats.jpg"
OUTPUT_DIR = Path(__file__).parent / "output"

skip_model_tests = pytest.mark.skipif(
    os.environ.get("GS2_SKIP_MODEL_TESTS") == "1",
    reason="GS2_SKIP_MODEL_TESTS=1",
)


class TestBuildPrompt:
    def test_lowercases_and_joins_with_periods(self):
        assert (
            pipeline.build_prompt(["Black Plastic Bracket", "HEX BOLT"])
            == "black plastic bracket. hex bolt."
        )

    def test_strips_existing_periods_and_whitespace(self):
        assert pipeline.build_prompt(["  a cat. ", "dog"]) == "a cat. dog."

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pipeline.build_prompt(["", "   "])


class TestPipelineConstruction:
    def test_ctor_does_not_load_models(self):
        """Models must lazy-load on first inference, not at construction."""
        p = pipeline.GroundedSam2Pipeline()
        assert p._dino_model is None
        assert p._sam_backend is None

    def test_unload_resets(self):
        p = pipeline.GroundedSam2Pipeline()
        p._dino_model = object()
        p._sam_backend = object()
        p.unload()
        assert p._dino_model is None
        assert p._sam_backend is None


@skip_model_tests
class TestOnFixtureImage:
    @pytest.fixture(scope="class")
    def pipe(self):
        return pipeline.GroundedSam2Pipeline()

    @pytest.fixture(scope="class")
    def image(self):
        return Image.open(FIXTURE)

    def test_detect_finds_cats(self, pipe, image):
        detections = pipe.detect(image, ["a cat", "a remote control"])
        assert len(detections) >= 2
        cat_dets = [d for d in detections if d.query == "a cat"]
        assert len(cat_dets) >= 2, f"expected 2+ cats, got {detections}"
        for d in detections:
            assert 0.0 < d.score <= 1.0
            x_min, y_min, x_max, y_max = d.box_xyxy
            assert 0 <= x_min < x_max <= image.width
            assert 0 <= y_min < y_max <= image.height

    def test_segment_produces_masks_and_overlay(self, pipe, image):
        segments = pipe.segment(image, ["a cat"])
        assert len(segments) >= 2
        for seg in segments:
            assert seg.mask.shape == (image.height, image.width)
            assert seg.mask.dtype == bool
            assert seg.mask.sum() >= pipe.min_mask_area_px
            # mask should mostly live inside its own box
            x_min, y_min, x_max, y_max = (int(v) for v in seg.box_xyxy)
            inside = seg.mask[y_min:y_max, x_min:x_max].sum()
            assert inside / seg.mask.sum() > 0.9

        # save an overlay PNG for eyeballing (milestone 1)
        OUTPUT_DIR.mkdir(exist_ok=True)
        overlay = np.array(image.convert("RGB"), dtype=np.float32)
        colors = [(255, 60, 60), (60, 255, 60), (60, 60, 255), (255, 255, 60)]
        for i, seg in enumerate(segments):
            color = np.array(colors[i % len(colors)], dtype=np.float32)
            overlay[seg.mask] = 0.5 * overlay[seg.mask] + 0.5 * color
        out_path = OUTPUT_DIR / "overlay.png"
        Image.fromarray(overlay.astype(np.uint8)).save(out_path)
        print(f"\noverlay saved to {out_path}")

    def test_per_call_threshold_override(self, pipe, image):
        strict = pipe.detect(image, ["a cat"], box_threshold=0.99)
        assert strict == []
