"""Command-line entrypoint for synthetic image generation."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from visscore_synthetic.pipelines import (
    FLUX_MODEL_ID,
    SDXL_MODEL_ID,
    get_pipeline_and_generator,
    snap_multiple,
)
from visscore_synthetic.prompts import NON_TUFE_PROMPTS, TUFE_PROMPTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Tufte-style vs chartjunk synthetic images via Diffusers (GPU).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Output root; creates tufte/ and non_tufte/ subfolders",
    )
    p.add_argument("--n-tufte", type=int, default=150, help="Number of Tufte-aligned images")
    p.add_argument("--n-non-tufte", type=int, default=150, help="Number of non-Tufte images")
    p.add_argument(
        "--model",
        choices=["flux-schnell", "sdxl"],
        default="flux-schnell",
        help="flux-schnell (default) or sdxl for lower VRAM",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=768,
        help="Square output side in pixels (snapped down to multiple of 8)",
    )
    p.add_argument("--seed", type=int, default=42, help="Base seed; per-image seeds derived from this")
    p.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token (else HF_TOKEN / HUGGING_FACE_HUB_TOKEN)",
    )
    p.add_argument(
        "--flux-model-id",
        default=FLUX_MODEL_ID,
        help="Override FLUX checkpoint id",
    )
    p.add_argument(
        "--sdxl-model-id",
        default=SDXL_MODEL_ID,
        help="Override SDXL checkpoint id",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root: Path = args.out_dir.resolve()
    tu_dir = out_root / "tufte"
    nt_dir = out_root / "non_tufte"
    tu_dir.mkdir(parents=True, exist_ok=True)
    nt_dir.mkdir(parents=True, exist_ok=True)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if args.model == "flux-schnell" and not hf_token:
        logger.warning(
            "No HF token provided. Gated FLUX weights may fail to download. "
            "Set HF_TOKEN or pass --hf-token after accepting the model license on Hugging Face.",
        )

    h = w = snap_multiple(args.image_size)

    try:
        pipe, gen = get_pipeline_and_generator(
            args.model,
            args.flux_model_id,
            args.sdxl_model_id,
            hf_token,
        )
    except Exception as e:
        logger.error("Failed to load pipeline: %s", e)
        if args.model == "flux-schnell":
            logger.error(
                "If this was access denied, accept the FLUX license on Hugging Face and set HF_TOKEN, "
                "or retry with: --model sdxl --image-size 512",
            )
        return 1

    base = args.seed

    for i in range(args.n_tufte):
        prompt = TUFE_PROMPTS[i % len(TUFE_PROMPTS)]
        path = tu_dir / f"tufte_{i:04d}.png"
        seed = base + i
        try:
            logger.info("Tufte %s/%s seed=%s", i + 1, args.n_tufte, seed)
            img = gen(pipe, prompt, h, w, seed)
            img.save(path)
        except RuntimeError as e:
            err = str(e).lower()
            if "out of memory" in err or ("cuda" in err and "alloc" in err):
                logger.error(
                    "CUDA out of memory. Try: --model sdxl --image-size 512 "
                    "(and restart runtime to free VRAM).",
                )
            raise

    offset = 10_000
    for i in range(args.n_non_tufte):
        prompt = NON_TUFE_PROMPTS[i % len(NON_TUFE_PROMPTS)]
        path = nt_dir / f"non_tufte_{i:04d}.png"
        seed = base + offset + i
        try:
            logger.info("Non-Tufte %s/%s seed=%s", i + 1, args.n_non_tufte, seed)
            img = gen(pipe, prompt, h, w, seed)
            img.save(path)
        except RuntimeError as e:
            err = str(e).lower()
            if "out of memory" in err or ("cuda" in err and "alloc" in err):
                logger.error(
                    "CUDA out of memory. Try: --model sdxl --image-size 512 "
                    "(and restart runtime to free VRAM).",
                )
            raise

    logger.info("Done. Wrote %s Tufte and %s non-Tufte images under %s", args.n_tufte, args.n_non_tufte, out_root)
    return 0


def entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
