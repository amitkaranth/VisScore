"""Diffusers pipeline loading and single-image generation."""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def snap_multiple(n: int, m: int = 8) -> int:
    n = max(m, n)
    return (n // m) * m


def _pick_torch_dtype_for_flux() -> Any:
    import torch

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _pick_torch_dtype_for_sdxl() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_flux(model_id: str, hf_token: str | None) -> Any:
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


def load_sdxl(model_id: str, hf_token: str | None) -> Any:
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


def generate_flux(
    pipe: Any,
    prompt: str,
    height: int,
    width: int,
    seed: int,
) -> Any:
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


def generate_sdxl(
    pipe: Any,
    prompt: str,
    height: int,
    width: int,
    seed: int,
) -> Any:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    negative = prompt.split("NEGATIVE:")[-1] if "NEGATIVE:" in prompt else None
    positive = prompt.split("NEGATIVE:")[0]

    out = pipe(
        positive,
        negative_prompt=negative,
        height=height,
        width=width,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    )


    return out.images[0]


def get_pipeline_and_generator(
    model: str,
    flux_model_id: str,
    sdxl_model_id: str,
    hf_token: str | None,
) -> tuple[Any, Callable[..., Any]]:
    if model == "flux-schnell":
        logger.info("Loading %s (this may take several minutes on first run)...", flux_model_id)
        return load_flux(flux_model_id, hf_token), generate_flux
    logger.info("Loading %s ...", sdxl_model_id)
    return load_sdxl(sdxl_model_id, hf_token), generate_sdxl
