"""
VisScore: CNN-Based Visualization Quality Classifier
Transfer learning with ResNet-50 / EfficientNet-B0 for binary classification
of chart images as good or bad based on Tufte's design principles.

Features:
  - Transfer learning with pretrained ImageNet weights
  - Data augmentation for small datasets
  - Grad-CAM visualization for model interpretability
  - Training/validation curves and confusion matrix
  - Supports both synthetic and real-world datasets

Setup:
  pip install torch torchvision matplotlib scikit-learn Pillow tqdm

Usage:
  python cnn_training.py --data_dir ./vis_dataset --epochs 25 --model resnet50
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)


# ============================================================
# DATASET
# ============================================================

class VisDataset(Dataset):
    """Dataset for loading visualization images with good/bad labels."""

    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.labels = []

        # Load from directory structure: data_dir/good/*.png, data_dir/bad/*.png
        for label_name, label_val in [("good", 1), ("bad", 0)]:
            label_dir = os.path.join(data_dir, label_name)
            if not os.path.exists(label_dir):
                print(f"Warning: {label_dir} not found, skipping.")
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(os.path.join(label_dir, fname))
                    self.labels.append(label_val)

        print(f"Loaded {len(self.samples)} images "
              f"(good={sum(self.labels)}, bad={len(self.labels)-sum(self.labels)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# TRANSFORMS
# ============================================================

def get_transforms(
    img_size=224,
    aug_strength="default",
    random_erasing_p=0.0,
    rotation_degrees=0.0,
):
    """Get training and validation transforms."""
    if aug_strength not in {"none", "default", "strong"}:
        raise ValueError("aug_strength must be one of: none, default, strong")

    if aug_strength == "none":
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return train_tf, val_tf

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=rotation_degrees) if rotation_degrees and rotation_degrees > 0 else transforms.Lambda(lambda x: x),
        transforms.ColorJitter(
            brightness=0.2 if aug_strength == "default" else 0.35,
            contrast=0.2 if aug_strength == "default" else 0.35,
            saturation=0.1 if aug_strength == "default" else 0.25,
            hue=0.0 if aug_strength == "default" else 0.03,
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(p=random_erasing_p) if random_erasing_p and random_erasing_p > 0 else transforms.Lambda(lambda x: x),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ============================================================
# MODEL
# ============================================================

def build_model(
    model_name="resnet50",
    num_classes=1,
    backbone_train_mode="last_block",
    head_dropout1=0.3,
    head_dropout2=0.2,
):
    """Build a transfer learning model for binary classification."""
    if backbone_train_mode not in {"head_only", "last_block", "full"}:
        raise ValueError("backbone_train_mode must be one of: head_only, last_block, full")

    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        if backbone_train_mode in {"head_only", "last_block"}:
            for param in model.parameters():
                param.requires_grad = False
            if backbone_train_mode == "last_block":
                for param in model.layer4.parameters():
                    param.requires_grad = True
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(head_dropout1),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout2),
            nn.Linear(256, num_classes),
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        if backbone_train_mode in {"head_only", "last_block"}:
            for param in model.parameters():
                param.requires_grad = False
            if backbone_train_mode == "last_block":
                for param in model.features[-2:].parameters():
                    param.requires_grad = True
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(head_dropout1),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout2),
            nn.Linear(256, num_classes),
        )

    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        if backbone_train_mode in {"head_only", "last_block"}:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(head_dropout1),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout2),
            nn.Linear(256, num_classes),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Trainable: {trainable:,} / {total:,} params")
    return model


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            # Keep activations attached to graph so we can differentiate w.r.t. them
            self.activations = out

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        # Ensure autograd tracks computations even if backbone weights are frozen.
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)  # shape [B, 1]

        # For binary classification, use the logit as the target score.
        score = output[:, 0].sum()

        # Compute gradients explicitly w.r.t. the hooked activations.
        if self.activations is None:
            raise RuntimeError("Grad-CAM activations were not captured. Check target layer selection.")

        grads = torch.autograd.grad(score, self.activations, retain_graph=False, create_graph=False)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def visualize_gradcam(model, dataset, device, model_name, output_dir, n=8):
    """Generate Grad-CAM visualizations for sample images."""
    # Get target layer based on model architecture
    if "resnet" in model_name:
        target_layer = model.layer4[-1]
    elif "efficientnet" in model_name:
        target_layer = model.features[-1]
    elif "vgg" in model_name:
        target_layer = model.features[-1]
    else:
        return

    gradcam = GradCAM(model, target_layer)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)

    for col, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        inp = img_tensor.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            pred = torch.sigmoid(model(inp)).item()

        # Generate Grad-CAM
        cam = gradcam.generate(inp)

        # De-normalize image for display
        img_display = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)

        # Resize CAM to image size
        import cv2
        cam_resized = cv2.resize(cam, (img_display.shape[1], img_display.shape[0]))

        # Original image
        axes[0, col].imshow(img_display)
        truth = "Good" if label == 1 else "Bad"
        pred_label = "Good" if pred > 0.5 else "Bad"
        color = "green" if truth == pred_label else "red"
        axes[0, col].set_title(f"True: {truth}\nPred: {pred_label} ({pred:.2f})",
                               fontsize=8, color=color)
        axes[0, col].axis('off')

        # Grad-CAM overlay
        axes[1, col].imshow(img_display)
        axes[1, col].imshow(cam_resized, cmap='jet', alpha=0.5)
        axes[1, col].set_title("Grad-CAM", fontsize=8)
        axes[1, col].axis('off')

    plt.suptitle("Grad-CAM: What the Model Looks At", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradcam_results.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM saved to {output_dir}/gradcam_results.png")


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            outputs = model(imgs).squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_probs), np.array(all_labels)


# ============================================================
# PLOTTING
# ============================================================

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_evaluation(probs, labels, output_dir):
    """Plot confusion matrix, ROC curve, and precision-recall curve."""
    preds = (probs > 0.5).astype(int)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Bad', 'Good'])
    ax1.set_yticklabels(['Bad', 'Good'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha='center', va='center',
                     fontsize=18, color='white' if cm[i, j] > cm.max()/2 else 'black')

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    ax3.plot(recall, precision, 'g-', linewidth=2)
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Bad", "Good"]))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train VisScore CNN classifier")
    parser.add_argument("--data_dir", type=str, default="./vis_dataset",
                        help="Dataset directory with good/ and bad/ subdirs")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save model and results")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "vgg16"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 regularization (AdamW-style) to reduce overfitting")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--aug_strength", type=str, default="default",
                        choices=["none", "default", "strong"],
                        help="Augmentation strength for training transforms")
    parser.add_argument("--random_erasing_p", type=float, default=0.1,
                        help="RandomErasing probability (0 to disable)")
    parser.add_argument("--rotation_degrees", type=float, default=5.0,
                        help="RandomRotation degrees (0 to disable)")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Fraction for validation set")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Fraction for held-out test set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone_train_mode", type=str, default="head_only",
                        choices=["head_only", "last_block", "full"],
                        help="How much of the pretrained backbone to fine-tune")
    parser.add_argument("--head_dropout1", type=float, default=0.5)
    parser.add_argument("--head_dropout2", type=float, default=0.5)
    parser.add_argument("--save_best", type=str, default="val_loss",
                        choices=["val_loss", "val_acc"],
                        help="Which metric to use for checkpointing best_model.pth")
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="Stop if best metric doesn't improve after N epochs (0 disables)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data — 70/15/15 train/val/test split
    train_tf, val_tf = get_transforms(
        img_size=args.img_size,
        aug_strength=args.aug_strength,
        random_erasing_p=args.random_erasing_p,
        rotation_degrees=args.rotation_degrees,
    )
    full_dataset = VisDataset(args.data_dir, transform=train_tf)
    
    test_size = int(len(full_dataset) * args.test_split)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size - test_size
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    # Override transform for validation and test sets (no augmentation)
    val_dataset_wrapper = VisDataset(args.data_dir, transform=val_tf)
    val_ds_proper = torch.utils.data.Subset(val_dataset_wrapper, val_ds.indices)
    test_ds_proper = torch.utils.data.Subset(val_dataset_wrapper, test_ds.indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds_proper, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds_proper, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds_proper)} | Test: {len(test_ds_proper)}")

    # Model
    model = build_model(
        args.model,
        num_classes=1,
        backbone_train_mode=args.backbone_train_mode,
        head_dropout1=args.head_dropout1,
        head_dropout2=args.head_dropout2,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = -float("inf")
    best_val_loss = float("inf")
    epochs_without_improve = 0

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_probs, v_labels = evaluate(model, val_loader, criterion, device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        scheduler.step(v_loss)

        print(f"  Epoch {epoch:2d}/{args.epochs} | "
              f"Train Loss: {t_loss:.4f} Acc: {t_acc:.3f} | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.3f}")

        improved = False
        if args.save_best == "val_loss":
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                improved = True
        else:
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                improved = True

        if improved:
            epochs_without_improve = 0
            best_val_acc = max(best_val_acc, v_acc)
            best_val_loss = min(best_val_loss, v_loss)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
        else:
            epochs_without_improve += 1

        if args.early_stop_patience and args.early_stop_patience > 0:
            if epochs_without_improve >= args.early_stop_patience:
                print(f"Early stopping: no improvement in {epochs_without_improve} epochs "
                      f"(save_best={args.save_best}).")
                break

    # Final evaluation on HELD-OUT TEST SET
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
    _, final_acc, final_probs, final_labels = evaluate(model, test_loader, criterion, device)

    if best_val_loss < float("inf"):
        print(f"\nBest Validation Loss: {best_val_loss:.4f}")
    if best_val_acc > -float("inf"):
        print(f"Best Validation Accuracy: {best_val_acc:.3f}")
    print(f"Test Set Accuracy: {final_acc:.3f}")

    # Generate plots
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, args.output_dir)
    plot_evaluation(final_probs, final_labels, args.output_dir)
    print(f"Training curves saved to {args.output_dir}/training_curves.png")
    print(f"Evaluation results saved to {args.output_dir}/evaluation_results.png")

    # Save training config
    config = vars(args)
    config["best_val_accuracy"] = best_val_acc
    config["best_val_loss"] = best_val_loss
    config["final_val_accuracy"] = final_acc
    config["device"] = str(device)
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAll results saved to {args.output_dir}/")

    # Grad-CAM visualizations (non-fatal; runs after saving config)
    try:
        import cv2
        visualize_gradcam(model, val_dataset_wrapper, device, args.model, args.output_dir, n=8)
    except ImportError:
        print("Install opencv-python for Grad-CAM: pip install opencv-python")
    except Exception as e:
        print(f"Grad-CAM skipped due to error: {e}")


if __name__ == "__main__":
    main()