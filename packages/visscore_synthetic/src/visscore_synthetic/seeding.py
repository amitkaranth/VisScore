"""Deterministic RNG streams per dataset row."""

from __future__ import annotations

import numpy as np

_CLASS_CODE = {"tufte": 0, "non_tufte": 1}


def image_rng(global_seed: int, class_label: str, index: int) -> np.random.Generator:
    """
    Stable across Python runs (no str.hash). Same (global_seed, class_label, index)
    always yields the same stream.
    """
    code = _CLASS_CODE[class_label]
    ss = np.random.SeedSequence([int(global_seed) & 0xFFFFFFFF, code, int(index) & 0xFFFFFFFF])
    return np.random.default_rng(ss)


def augment_subrng(global_seed: int, class_label: str, index: int) -> np.random.Generator:
    """Separate stream so augmentation does not consume chart RNG ordering."""
    code = _CLASS_CODE[class_label]
    ss = np.random.SeedSequence(
        [int(global_seed) & 0xFFFFFFFF, code, int(index) & 0xFFFFFFFF, 0xA051EED]
    )
    return np.random.default_rng(ss)
