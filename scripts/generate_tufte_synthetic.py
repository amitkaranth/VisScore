#!/usr/bin/env python3
"""
Generate synthetic chart-style images for binary CNN training: Tufte-aligned vs chartjunk.

Runs on Google Colab (GPU) or locally with CUDA. Uses Hugging Face Diffusers and open
weights — no paid image APIs.

Models:
  - flux-schnell (default): black-forest-labs/FLUX.1-schnell (gated; free HF account + token).
  - sdxl: stabilityai/stable-diffusion-xl-base-1.0 (lower VRAM fallback).

Environment:
  HF_TOKEN or HUGGING_FACE_HUB_TOKEN for gated / licensed downloads.

Example:
  export HF_TOKEN=hf_...
  python scripts/generate_tufte_synthetic.py --out-dir data/synthetic --model sdxl --image-size 512
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

TUFE_PROMPTS: list[str] = [
    "Clean minimalist line chart, Edward Tufte style, high data-ink ratio, faint or no gridlines, "
    "no chartjunk, flat 2D only, small sans-serif axis labels, muted neutral colors, white background, "
    "professional editorial statistics graphic",
    "Sparse scatter plot, Tufte-inspired design, maximum data emphasis, thin axes, no decorative borders, "
    "no 3D, restrained grayscale and one accent color, plenty of whitespace, scientific publication quality",
    "Simple bar chart, minimalist data visualization, high data-ink, subtle tick marks only, "
    "no icons or clipart, flat colors, clear typographic hierarchy, light gray grid optional, white paper",
    "Small multiples panel of tiny line charts, consistent scales, Tufte small-multiples style, "
    "dense data sparse ink, no drop shadows, no gradients, monochrome with minimal ink",
    "Horizontal bar ranking chart, clean Tufte aesthetic, direct labeling where possible, "
    "no heavy frames, 2D flat, neutral palette, institutional report style",
    "Minimal area chart with thin stroke, white background, understated axes, no legends box if avoidable, "
    "Edward Tufte principles, no embellishment, high clarity",
    "Dot plot / strip plot style visualization, simple geometry, thin rules, small text, "
    "no neon, no 3D pie, serious analytic look",
    "Sparkline row beneath small table, Tufte sparkline style, tiny high-resolution trend lines, "
    "gray baseline only, no chart decoration",
    "Dual-axis avoided; single clear line chart with modest grid, Tufte data-ink focus, "
    "sans-serif captions, calm colors, print-ready infographic",
    "Column chart with narrow bars, white space between groups, minimal axis lines, "
    "no gradients or glow, flat 2D, textbook Tufte compliance",
    "Time series with light vertical reference lines only, sparse ink, readable micro-labels, "
    "no pictograms, neutral tones, executive briefing chart",
    "Box plot summary, minimal ink, thin whiskers, small median marks, gray axes on white, "
    "no 3D, academic Tufte-style figure",
]

NON_TUFE_PROMPTS: list[str] = [
    "Extremely busy dashboard chart, heavy chartjunk, thick glowing borders, neon cyan and magenta, "
    "3D extruded bar chart, lens flare style highlights, decorative icons, dark gamer UI chrome",
    "Cluttered pie chart with exploded slices, drop shadows, bevel and emboss, saturated rainbow colors, "
    "heavy gradient fills, clipart dollar signs, 3D perspective, misleading emphasis",
    "Overdesigned infographic bar chart, skeuomorphic metal textures, unnecessary 3D rotation, "
    "busy background pattern, huge decorative title banner, chartjunk everywhere",
    "Flashy line chart with thick gradient area fill, glowing neon grid, starburst icons, "
    "heavy drop shadow under plot, busy textured backdrop, cyberpunk dashboard widget",
    "3D column chart with fake depth and perspective distortion, rainbow color cycle per bar, "
    "thick white outlines, cartoon mascot in corner, cluttered legend boxes",
    "Messy combo chart, too many y-axis colors, decorative arrows and callouts, "
    "heavy frame with rounded corners, glossy glass effect, unnecessary pictograms",
    "Pie chart with 12 rainbow slices, 3D tilt, exploded segments, shadows, "
    "sparkle effects, busy legend with icons, poster-style clutter",
    "Dark mode chart with loud gradients, animated-style glow on bars, hex grid background noise, "
    "futuristic HUD ornaments, low data-ink flashy visualization",
    "Stock-photo style infographic chart, watermarks, huge 3D numbers floating, "
    "clipart people pointing, excessive arrows, neon outlines",
    "Dashboard tile with fake leather texture, embossed buttons, 3D donut charts, "
    "busy KPI badges, chartjunk borders and stickers",
    "Overlapping translucent layers, misaligned 3D bars, rainbow laser gradients, "
    "decorative corner flourishes, unreadable tiny 3D text",
    "Skeuomorphic thermometer chart, glossy plastic tubes, unnecessary 3D, "
    "lens reflections, busy wallpaper background, comic sans style clutter",
]


def _snap_multiple(n: int, m: int = 8) -> int:
    n = max(m, n)
    return (n // m) * m


def _pick_torch_dtype_for_flux() -> "torch.dtype":
    import torch

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _pick_torch_dtype_for_sdxl() -> "torch.dtype":
    import torch

    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _load_flux(model_id: str, hf_token: str | None) -> object:
    import torch
    from diffusers import FluxPipeline

    dtype = _pick_torch_dtype_for_flux()
    kwargs: dict = {"torch_dtype": dtype}
    if hf_token:
        kwargs["token"] = hf_token

    pipe = FluxPipeline.from_pretrained(model_id, **kwargs)
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    else:
        logger.warning("No CUDA; FLUX on CPU will be extremely slow.")
        pipe.to("cpu")
    return pipe


def _load_sdxl(model_id: str, hf_token: str | None) -> object:
    import torch
    from diffusers import StableDiffusionXLPipeline

    dtype = _pick_torch_dtype_for_sdxl()
    kwargs: dict = {"torch_dtype": dtype}
    if hf_token:
        kwargs["token"] = hf_token

    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **kwargs)
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    else:
        logger.warning("No CUDA; SDXL on CPU will be very slow.")
        pipe.to("cpu")
    return pipe


def _generate_flux(
    pipe: object,
    prompt: str,
    height: int,
    width: int,
    seed: int,
) -> "object":
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        height=height,
        width=width,
        generator=generator,
    )
    return out.images[0]


def _generate_sdxl(
    pipe: object,
    prompt: str,
    height: int,
    width: int,
    seed: int,
) -> "object":
    import torch

    # CPU generator matches enable_model_cpu_offload() and avoids device mismatch issues.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=30,
        guidance_scale=7.0,
        generator=generator,
    )
    return out.images[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Tufte-style vs chartjunk synthetic images via Diffusers (Colab/local GPU).",
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
    return p.parse_args()


def main() -> int:
    args = parse_args()
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

    h = w = _snap_multiple(args.image_size)

    try:
        if args.model == "flux-schnell":
            logger.info("Loading %s (this may take several minutes on first run)...", args.flux_model_id)
            pipe = _load_flux(args.flux_model_id, hf_token)
            gen = _generate_flux
        else:
            logger.info("Loading %s ...", args.sdxl_model_id)
            pipe = _load_sdxl(args.sdxl_model_id, hf_token)
            gen = _generate_sdxl
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
            if "out of memory" in err or "cuda" in err and "alloc" in err:
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
            if "out of memory" in err or "cuda" in err and "alloc" in err:
                logger.error(
                    "CUDA out of memory. Try: --model sdxl --image-size 512 "
                    "(and restart runtime to free VRAM).",
                )
            raise

    logger.info("Done. Wrote %s Tufte and %s non-Tufte images under %s", args.n_tufte, args.n_non_tufte, out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
