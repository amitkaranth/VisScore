"""
VisScore: Standalone Inference Script
Feed any chart image and get a quality prediction + Grad-CAM heatmap.

Usage:
  # Single image
  python inference.py --image chart.png --model_path ./results/best_model.pth

  # Entire folder of images
  python inference.py --image_dir ./test_charts/ --model_path ./results/best_model.pth

  # With Grad-CAM visualization saved
  python inference.py --image chart.png --model_path ./results/best_model.pth --gradcam
"""

import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# MODEL (must match training architecture exactly)
# ============================================================

def load_model(model_path, model_name="resnet50", device="cpu"):
    """Load trained model from checkpoint."""
    if model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=False)
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


# ============================================================
# PREPROCESSING (must match training transforms)
# ============================================================

def preprocess_image(image_path, img_size=224):
    """Load and preprocess a single image for inference."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]
    return tensor, img


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach())
        )
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach())
        )

    def generate(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def save_gradcam(model, model_name, input_tensor, original_img, output_path, prediction):
    """Generate and save Grad-CAM overlay."""
    try:
        import cv2
    except ImportError:
        print("  Skipping Grad-CAM (install opencv-python: pip install opencv-python)")
        return

    # Get target layer
    if "resnet" in model_name:
        target_layer = model.layer4[-1]
    elif "efficientnet" in model_name:
        target_layer = model.features[-1]
    elif "vgg" in model_name:
        target_layer = model.features[-1]
    else:
        return

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor)

    # Resize CAM to original image size
    img_np = np.array(original_img.resize((224, 224))) / 255.0
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    ax1.imshow(img_np)
    ax1.set_title("Original Chart", fontsize=12)
    ax1.axis('off')

    # Grad-CAM heatmap
    ax2.imshow(cam_resized, cmap='jet')
    ax2.set_title("Grad-CAM Heatmap", fontsize=12)
    ax2.axis('off')

    # Overlay
    ax3.imshow(img_np)
    ax3.imshow(cam_resized, cmap='jet', alpha=0.5)
    label = prediction["label"]
    prob = prediction["probability"]
    color = "green" if label == "GOOD" else "red"
    ax3.set_title(f"Prediction: {label} ({prob:.1%})", fontsize=12, color=color, fontweight='bold')
    ax3.axis('off')

    plt.suptitle("VisScore: Visualization Quality Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grad-CAM saved to {output_path}")


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image_path, device):
    """Run inference on a single image and return prediction."""
    input_tensor, original_img = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logit = model(input_tensor).squeeze()
        probability = torch.sigmoid(logit).item()

    label = "GOOD" if probability > 0.6 else "BAD"
    confidence = probability if label == "GOOD" else (1 - probability)

    result = {
        "image": os.path.basename(image_path),
        "label": label,
        "probability": probability,
        "confidence": confidence,
    }
    return result, input_tensor, original_img


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VisScore: Predict visualization quality")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single chart image")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path to a folder of chart images")
    parser.add_argument("--model_path", type=str, default="./results/best_model.pth",
                        help="Path to trained model weights")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "vgg16"])
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM visualizations")
    parser.add_argument("--output_dir", type=str, default="./predictions",
                        help="Directory to save Grad-CAM images and results")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        print("Error: Provide --image or --image_dir")
        print("Example: python inference.py --image chart.png --model_path ./results/best_model.pth")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args.model_name, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(args.image_dir, f))

    if not image_paths:
        print("No valid images found.")
        return

    # Run predictions
    all_results = []
    print(f"\nAnalyzing {len(image_paths)} image(s)...\n")
    print(f"{'Image':<40} {'Label':<8} {'Probability':<12} {'Confidence':<12}")
    print("-" * 72)

    for img_path in image_paths:
        result, input_tensor, original_img = predict(model, img_path, device)
        all_results.append(result)

        # Print result
        label_color = "\033[92m" if result["label"] == "GOOD" else "\033[91m"
        reset = "\033[0m"
        print(f"{result['image']:<40} {label_color}{result['label']:<8}{reset} "
              f"{result['probability']:<12.4f} {result['confidence']:<12.1%}")

        # Generate Grad-CAM if requested
        if args.gradcam:
            cam_path = os.path.join(
                args.output_dir,
                f"gradcam_{os.path.splitext(result['image'])[0]}.png"
            )
            save_gradcam(model, args.model_name, input_tensor, original_img,
                         cam_path, result)

    # Save results JSON
    results_path = os.path.join(args.output_dir, "predictions.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    good_count = sum(1 for r in all_results if r["label"] == "GOOD")
    bad_count = len(all_results) - good_count
    avg_prob = np.mean([r["probability"] for r in all_results])

    print(f"\n{'=' * 72}")
    print(f"Summary: {good_count} GOOD | {bad_count} BAD | Avg Score: {avg_prob:.3f}")
    print(f"Results saved to {results_path}")
    if args.gradcam:
        print(f"Grad-CAM images saved to {args.output_dir}/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()