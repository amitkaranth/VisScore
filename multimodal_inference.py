#!/usr/bin/env python3
"""
VisScore: Multimodal inference — CNN + VLM (Tufte reasoning) + consensus.

**Gemini** (default): https://aistudio.google.com/apikey — GEMINI_API_KEY
  Tries your --vlm_model first, then aliases like gemini-flash-latest if you get 404.

**Groq** (free): https://console.groq.com/keys — GROQ_API_KEY
  python multimodal_inference.py --image ./inference/image_bad.png --vlm_provider groq

Usage:
  python multimodal_inference.py --image ./inference/image_bad.png
  python multimodal_inference.py --image chart.png --vlm_provider groq --gradcam

Optional env: VLM_PROVIDER, GEMINI_VLM_MODEL, GROQ_VLM_MODEL
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import load_model, predict, save_gradcam
from vlm_judge import combine_cnn_and_vlm, judge_chart_vlm

import torch


def run_one(
    image_path: str,
    model,
    model_name: str,
    device,
    vlm_provider: str,
    gemini_api_key: str | None,
    groq_api_key: str | None,
    vlm_model: str | None,
    gradcam: bool,
    output_dir: str,
) -> dict:
    result, input_tensor, original_img = predict(model, image_path, device)
    vlm = judge_chart_vlm(
        image_path,
        provider=vlm_provider,
        gemini_api_key=gemini_api_key,
        groq_api_key=groq_api_key,
        model_name=vlm_model,
    )
    consensus = combine_cnn_and_vlm(result["label"], vlm)

    out = {
        "image": os.path.basename(image_path),
        "vlm_provider": vlm_provider,
        "cnn": {
            "label": result["label"],
            "probability": result["probability"],
            "confidence": result["confidence"],
        },
        "vlm": {
            "verdict": vlm.get("verdict"),
            "reasoning": vlm.get("reasoning", ""),
            "model": vlm.get("model"),
            "error": vlm.get("error"),
        },
        "consensus": consensus,
    }

    if gradcam:
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(result["image"])[0]
        cam_path = os.path.join(output_dir, f"gradcam_mm_{stem}.png")
        save_gradcam(model, model_name, input_tensor, original_img, cam_path, result)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="VisScore multimodal: CNN + Gemini VLM + consensus"
    )
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="./results/best_model.pth")
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "vgg16"],
    )
    parser.add_argument(
        "--vlm_provider",
        type=str,
        default=None,
        choices=["gemini", "groq"],
        help="VLM backend (default: env VLM_PROVIDER or gemini)",
    )
    parser.add_argument("--gemini_api_key", type=str, default=None,
                        help="Override GEMINI_API_KEY for this run")
    parser.add_argument("--groq_api_key", type=str, default=None,
                        help="Override GROQ_API_KEY for this run")
    parser.add_argument("--vlm_model", type=str, default=None,
                        help="Override model id (Gemini or Groq vision model)")
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./predictions_multimodal")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        print("Error: provide --image or --image_dir")
        sys.exit(1)

    provider = (args.vlm_provider or os.environ.get("VLM_PROVIDER") or "gemini").strip().lower()
    if provider not in ("gemini", "groq"):
        provider = "gemini"

    gemini_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY"
    )
    groq_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")

    if provider == "gemini" and not gemini_key:
        print(
            "Warning: No Gemini API key. Set GEMINI_API_KEY or use --vlm_provider groq with GROQ_API_KEY.\n"
            "  Gemini: https://aistudio.google.com/apikey  |  Groq: https://console.groq.com/keys"
        )
    if provider == "groq" and not groq_key:
        print(
            "Warning: No Groq API key. Set GROQ_API_KEY (free): https://console.groq.com/keys"
        )

    print(f"VLM provider: {provider}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(args.model_path, args.model_name, device)

    paths = []
    if args.image:
        paths.append(args.image)
    if args.image_dir:
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(args.image_dir, f))

    if not paths:
        print("No images found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []

    print(f"\nProcessing {len(paths)} image(s)…\n")

    for p in paths:
        row = run_one(
            p,
            model,
            args.model_name,
            device,
            provider,
            gemini_key,
            groq_key,
            args.vlm_model,
            args.gradcam,
            args.output_dir,
        )
        all_rows.append(row)

        c = row["consensus"]
        vlm = row["vlm"]
        print(f"--- {row['image']} ---")
        print(f"  CNN:  {row['cnn']['label']}  (p_good={row['cnn']['probability']:.4f})")
        if vlm.get("error"):
            print(f"  VLM:  ERROR — {vlm['error']}")
        else:
            print(f"  VLM:  {vlm.get('verdict')}  ({vlm.get('model')})")
            print(f"  Why:  {vlm.get('reasoning', '')[:400]}{'…' if len(vlm.get('reasoning', '')) > 400 else ''}")
        print(f"  Final: {c.get('final_verdict')}  [{c.get('agreement')}]")
        print()

    out_path = os.path.join(args.output_dir, "multimodal_results.json")
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    print(f"Saved: {out_path}")

    # Quick stats
    unanimous = sum(1 for r in all_rows if r["consensus"].get("agreement") == "unanimous")
    split = sum(1 for r in all_rows if r["consensus"].get("final_verdict") == "SPLIT")
    print(f"Summary: unanimous={unanimous}  split={split}  total={len(all_rows)}")


if __name__ == "__main__":
    main()
