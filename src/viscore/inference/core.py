"""
VisScore: inference utilities (CNN model load/predict + Grad-CAM).

This module contains the reusable logic; CLI wrappers should live under `cli/`.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_model(model_path: str, model_name: str = "resnet50", device: str | torch.device = "cpu"):
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
    return model


def preprocess_image(image_path: str, img_size: int = 224) -> Tuple[torch.Tensor, Image.Image]:
    """Load and preprocess a single image for inference."""
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return tensor, img


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "activations", o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def save_gradcam(model, model_name: str, input_tensor: torch.Tensor, original_img: Image.Image, output_path: str, prediction: dict):
    """Generate and save Grad-CAM overlay."""
    try:
        import cv2
    except ImportError:
        return

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

    img_np = np.array(original_img.resize((224, 224))) / 255.0
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(img_np)
    ax1.set_title("Original Chart", fontsize=12)
    ax1.axis("off")

    ax2.imshow(cam_resized, cmap="jet")
    ax2.set_title("Grad-CAM Heatmap", fontsize=12)
    ax2.axis("off")

    ax3.imshow(img_np)
    ax3.imshow(cam_resized, cmap="jet", alpha=0.5)
    label = prediction["label"]
    prob = prediction["probability"]
    color = "green" if label == "GOOD" else "red"
    ax3.set_title(f"Prediction: {label} ({prob:.1%})", fontsize=12, color=color, fontweight="bold")
    ax3.axis("off")

    plt.suptitle("VisScore: Visualization Quality Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def predict(model, image_path: str, device: torch.device):
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

