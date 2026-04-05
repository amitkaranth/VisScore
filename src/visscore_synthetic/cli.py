"""CLI for programmatic Tufte vs non-Tufte synthetic chart dataset generation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from visscore_synthetic.metadata import MetadataWriter
from visscore_synthetic.pipelines import save_chart_image
from visscore_synthetic.registry import (
    draw_random_chart,
    parse_chart_filter,
    NON_TUFE_CHARTS,
    TUFE_CHARTS,
)
from visscore_synthetic.seeding import augment_subrng, image_rng

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate synthetic chart images (matplotlib): tufte vs non_tufte classes. "
            "Fully local; no diffusion or Hugging Face."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Output root (creates tufte/ and non_tufte/)",
    )
    p.add_argument("--n-tufte", type=int, default=150, help="Number of tufte-class images")
    p.add_argument("--n-non-tufte", type=int, default=150, help="Number of non_tufte-class images")
    p.add_argument("--seed", type=int, default=42, help="Global seed (reproducible dataset)")
    p.add_argument("--dpi", type=float, default=100.0, help="Matplotlib figure DPI")

    g = p.add_argument_group("image dimensions (random per image, from RNG)")
    g.add_argument("--min-width", type=int, default=480, help="Min width pixels")
    g.add_argument("--max-width", type=int, default=896, help="Max width pixels")
    g.add_argument("--min-height", type=int, default=360, help="Min height pixels")
    g.add_argument("--max-height", type=int, default=672, help="Max height pixels")
    g.add_argument(
        "--image-size",
        type=int,
        default=None,
        metavar="PX",
        help="If set, fixes square size (sets min/max width and height to this value)",
    )

    p.add_argument(
        "--tufte-charts",
        type=str,
        default=None,
        help="Comma-separated subset of tufte chart keys (see registry)",
    )
    p.add_argument(
        "--non-tufte-charts",
        type=str,
        default=None,
        help="Comma-separated subset of non_tufte chart keys",
    )

    p.add_argument(
        "--augment",
        action="store_true",
        help="Apply mild PIL augmentation after render",
    )
    p.add_argument(
        "--style-strength",
        type=float,
        default=0.35,
        help="Augmentation intensity in [0,1] when --augment is set",
    )

    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="JSONL path (default: <out-dir>/metadata.jsonl). Use empty string with --no-metadata",
    )
    p.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not write metadata JSONL",
    )
    return p.parse_args(argv)


def _resolve_dims(args: argparse.Namespace) -> tuple[int, int, int, int]:
    if args.image_size is not None:
        s = int(args.image_size)
        return s, s, s, s
    return int(args.min_width), int(args.max_width), int(args.min_height), int(args.max_height)


def _generate_split(
    *,
    class_label: str,
    count: int,
    out_subdir: Path,
    global_seed: int,
    min_w: int,
    max_w: int,
    min_h: int,
    max_h: int,
    dpi: float,
    allowed: frozenset[str] | None,
    augment: bool,
    style_strength: float,
    meta_writer: MetadataWriter | None,
) -> None:
    prefix = "tufte" if class_label == "tufte" else "non_tufte"
    for i in range(count):
        rng = image_rng(global_seed, class_label, i)
        w = int(rng.integers(min_w, max_w + 1))
        h = int(rng.integers(min_h, max_h + 1))
        w = max(64, w)
        h = max(64, h)
        fig, meta = draw_random_chart(rng, class_label, w, h, dpi, allowed)
        fname = f"{prefix}_s{global_seed}_i{i:04d}.png"
        path = out_subdir / fname
        aug_rng = augment_subrng(global_seed, class_label, i)
        extra = save_chart_image(fig, path, dpi, augment, aug_rng, style_strength)
        logger.info("%s %s/%s -> %s (%sx%s)", class_label, i + 1, count, fname, extra["width"], extra["height"])
        if meta_writer is not None:
            row = {
                "filename": fname,
                "class_label": class_label,
                "chart_type": meta.get("chart_type"),
                "chart_family": meta.get("chart_family"),
                "global_seed": global_seed,
                "index": i,
                "dpi": dpi,
                "requested_width": w,
                "requested_height": h,
                **{k: v for k, v in meta.items() if k not in ("chart_type", "chart_family", "class_label")},
                **extra,
            }
            meta_writer.write_row(row)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root: Path = args.out_dir.resolve()
    tu_dir = out_root / "tufte"
    nt_dir = out_root / "non_tufte"
    tu_dir.mkdir(parents=True, exist_ok=True)
    nt_dir.mkdir(parents=True, exist_ok=True)

    min_w, max_w, min_h, max_h = _resolve_dims(args)
    if min_w > max_w or min_h > max_h:
        logger.error("Invalid dimension bounds")
        return 1

    try:
        tufte_allowed = parse_chart_filter(args.tufte_charts, TUFE_CHARTS)
        non_allowed = parse_chart_filter(args.non_tufte_charts, NON_TUFE_CHARTS)
    except ValueError as e:
        logger.error("%s", e)
        return 1

    meta_path: Path | None = None
    if not args.no_metadata:
        meta_path = args.metadata if args.metadata is not None else out_root / "metadata.jsonl"

    if meta_path is not None:
        with MetadataWriter(meta_path) as mw:
            _generate_split(
                class_label="tufte",
                count=args.n_tufte,
                out_subdir=tu_dir,
                global_seed=args.seed,
                min_w=min_w,
                max_w=max_w,
                min_h=min_h,
                max_h=max_h,
                dpi=float(args.dpi),
                allowed=tufte_allowed,
                augment=bool(args.augment),
                style_strength=float(args.style_strength),
                meta_writer=mw,
            )
            _generate_split(
                class_label="non_tufte",
                count=args.n_non_tufte,
                out_subdir=nt_dir,
                global_seed=args.seed,
                min_w=min_w,
                max_w=max_w,
                min_h=min_h,
                max_h=max_h,
                dpi=float(args.dpi),
                allowed=non_allowed,
                augment=bool(args.augment),
                style_strength=float(args.style_strength),
                meta_writer=mw,
            )
    else:
        _generate_split(
            class_label="tufte",
            count=args.n_tufte,
            out_subdir=tu_dir,
            global_seed=args.seed,
            min_w=min_w,
            max_w=max_w,
            min_h=min_h,
            max_h=max_h,
            dpi=float(args.dpi),
            allowed=tufte_allowed,
            augment=bool(args.augment),
            style_strength=float(args.style_strength),
            meta_writer=None,
        )
        _generate_split(
            class_label="non_tufte",
            count=args.n_non_tufte,
            out_subdir=nt_dir,
            global_seed=args.seed,
            min_w=min_w,
            max_w=max_w,
            min_h=min_h,
            max_h=max_h,
            dpi=float(args.dpi),
            allowed=non_allowed,
            augment=bool(args.augment),
            style_strength=float(args.style_strength),
            meta_writer=None,
        )

    logger.info(
        "Done. Wrote %s tufte + %s non_tufte images under %s",
        args.n_tufte,
        args.n_non_tufte,
        out_root,
    )
    return 0


def entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
