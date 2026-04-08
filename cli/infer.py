#!/usr/bin/env python3
"""
VisScore CLI: CNN-only inference (wrapper around `viscore.inference`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _bootstrap_src

_bootstrap_src.bootstrap()

from viscore.inference import load_model, predict, save_gradcam


def main() -> None:
    parser = argparse.ArgumentParser(description="VisScore: Predict visualization quality")
    parser.add_argument("--image", type=str, default=None, help="Path to a single chart image")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to a folder of chart images")
    parser.add_argument("--model_path", type=str, default="./results/best_model.pth", help="Path to trained model weights")
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0", "vgg16"])
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visualizations")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Directory to save Grad-CAM images and results")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        print("Error: Provide --image or --image_dir")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args.model_name, device)

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths: list[str] = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(args.image_dir, f))

    if not image_paths:
        print("No valid images found.")
        return

    all_results = []
    for img_path in image_paths:
        result, input_tensor, original_img = predict(model, img_path, device)
        all_results.append(result)
        if args.gradcam:
            cam_path = os.path.join(args.output_dir, f"gradcam_{os.path.splitext(result['image'])[0]}.png")
            save_gradcam(model, args.model_name, input_tensor, original_img, cam_path, result)

    results_path = os.path.join(args.output_dir, "predictions.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()

