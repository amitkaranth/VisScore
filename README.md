# VisScore: CNN-Based Visualization Quality Classifier

A comprehensive machine learning system for assessing visualization quality based on Edward Tufte's design principles. This project generates synthetic chart datasets, trains deep learning models for binary classification (good/bad charts), and provides an interactive web interface for predictions with explainability visualizations.

## 📋 Project Overview

**Problem**: Automatically assess whether a data visualization follows best practices and design principles.

**Solution**: 
- Generate synthetic "good" and "bad" chart pairs based on Tufte's principles
- Train a CNN model (ResNet-50, EfficientNet-B0, or VGG-16) on these images
- Deploy via Streamlit for interactive predictions with Grad-CAM visual explanations

**Key Features**:
- ✅ Synthetic data generation with diverse chart types and violations
- ✅ Multiple neural network architectures with transfer learning
- ✅ Grad-CAM heatmap visualization for model interpretability
- ✅ Batch inference pipeline (single image or directory)
- ✅ Interactive Streamlit web UI with multi-model support
- ✅ Comprehensive metrics: confusion matrix, ROC curves, precision-recall

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to project directory
cd /./DataVisualisation/VisScore

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset

```bash
python synthetic_data_gen.py --output_dir ./vis_dataset --num_samples 500
```

### 3. Train a Model

```bash
python cnn_training.py --data_dir ./vis_dataset --model resnet50 --epochs 25
```

### 4. Run Inference

```bash
python inference.py --image chart.png --model_path ./results/best_model.pth --gradcam
```

### 5. Launch Interactive UI

```bash
streamlit run app.py
```
Visit `http://localhost:8501` in your browser.

### 6. Multimodal inference (CNN + free Gemini VLM)

Combine your trained CNN with a **vision-language model** (Google Gemini, free API key) for Tufte-style **reasoning** and a **consensus** label.

1. Get a key: [Google AI Studio](https://aistudio.google.com/apikey) (free tier).
2. Install: `pip install google-generativeai` (included in `requirements.txt`).
3. Export the key: `export GEMINI_API_KEY="your_key"` (or paste it in the Streamlit sidebar).

**CLI** (writes `predictions_multimodal/multimodal_results.json`):

```bash
export GEMINI_API_KEY="your_key"
python multimodal_inference.py --image ./inference/image_bad.png --model_name resnet50 --gradcam
```

**Streamlit:** enable **“VLM + consensus”** in the sidebar and enter the same API key.

**Consensus rules:** if CNN and VLM **agree**, that label wins. If they **disagree**, the final label is **SPLIT** (no majority with two voters)—review CNN metrics and VLM reasoning. If the API fails, the app falls back to **CNN only**.

If Gemini returns **404** (model renamed) or **quota** errors, the code tries other Gemini IDs automatically (`gemini-flash-latest`, etc.). Alternatively use **Groq** (free vision API): `export VLM_PROVIDER=groq GROQ_API_KEY=...` then `python multimodal_inference.py --image ... --vlm_provider groq`.

---

## 📁 Project Structure

```
VisScore/
├── synthetic_data_gen.py    # Generate synthetic chart dataset
├── synthetic_data_gen_plotly.py  # Optional Plotly/Kaleido charts (style diversity)
├── cnn_training.py          # Train CNN models
├── inference.py             # Batch inference pipeline
├── multimodal_inference.py  # CNN + Gemini VLM + consensus (CLI)
├── vlm_judge.py             # Gemini Tufte prompt + JSON parsing
├── app.py                   # Streamlit web UI
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── vis_dataset/             # Generated synthetic data (after step 1)
│   ├── good/                # High-quality charts
│   ├── bad/                 # Low-quality charts
│   └── metadata.json        # Chart descriptions and violations
│
├── results/                 # Trained models and results (after step 2)
│   ├── best_model.pth       # Best trained model weights
│   ├── training_config.json # Training parameters for reproducibility
│   ├── training_curves.png  # Loss/accuracy plots
│   ├── evaluation_results.png # Confusion matrix, ROC, PR curves
│   └── gradcam_results.png  # Grad-CAM visualization examples
│
├── predictions/             # Inference outputs (after step 3)
│   ├── results.json         # Prediction results for batch
│   └── gradcam_*.png        # Grad-CAM overlays
│
└── __pycache__/             # Python cache (ignore)
```

---

## 🔧 Detailed Workflow

### Step 1: Generate Synthetic Dataset

**Purpose**: Create balanced dataset of "good" (Tufte-compliant) and "bad" (violating principles) charts.

```bash
python synthetic_data_gen.py [OPTIONS]
```

**Options**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | `./vis_dataset` | Directory to save generated images |
| `--num_samples` | int | `500` | Number of good/bad chart pairs to generate |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--augment_strength` | choice | `medium` | Post-processing augmentation: `none`, `low`, `medium`, `high` |
| `--seed_aug` | int | `None` | Optional seed for augmentation (keeps augmentations reproducible) |

**Examples**:

```bash
# Generate 500 chart pairs with medium augmentation
python synthetic_data_gen.py --num_samples 500 --augment_strength medium

# Generate 1000 pairs with high augmentation, custom seed
python synthetic_data_gen.py --num_samples 1000 --augment_strength high --seed 123

# Minimal augmentation for larger dataset
python synthetic_data_gen.py --num_samples 2000 --augment_strength low --output_dir ./large_dataset
```

**Chart Types Generated**:
- **Good Charts**: Bar, horizontal bar, line, scatter, pie (5 categories max), histogram
- **Bad Charts**: Chartjunk, 3D effects, truncated axes, rainbow colors, poor data-ink ratio, dual-axis abuse, heavy gridlines

**Augmentations Applied** (based on strength):
- Rotation (±5-15°)
- Resize (95-105%)
- Brightness/contrast adjustments
- Gaussian blur
- Gaussian noise

**Output**: 
- `vis_dataset/good/` - 500+ chart images
- `vis_dataset/bad/` - 500+ chart images  
- `vis_dataset/metadata.json` - Descriptions and violation tags

---

### Step 2: Train CNN Model

**Purpose**: Train a deep learning model to classify charts as good/bad.

```bash
python cnn_training.py [OPTIONS]
```

**Options**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | `./vis_dataset` | Dataset directory containing `good/` and `bad/` subdirs |
| `--output_dir` | str | `./results` | Directory to save trained model and results |
| `--model` | choice | `resnet50` | Model architecture: `resnet50`, `efficientnet_b0`, `vgg16` |
| `--epochs` | int | `25` | Number of training epochs |
| `--batch_size` | int | `32` | Batch size for training |
| `--lr` | float | `1e-4` | Learning rate (Adam optimizer) |
| `--img_size` | int | `224` | Input image size (ImageNet standard) |
| `--val_split` | float | `0.15` | Validation set fraction (0.0-1.0) |
| `--test_split` | float | `0.15` | Test set fraction (0.0-1.0) |
| `--seed` | int | `42` | Random seed for reproducibility |

**Examples**:

```bash
# Train ResNet-50 with default settings
python cnn_training.py

# Train EfficientNet-B0 with custom hyperparameters
python cnn_training.py --model efficientnet_b0 --epochs 50 --batch_size 16 --lr 5e-5

# Train VGG-16 on custom dataset
python cnn_training.py --model vgg16 --data_dir ./my_charts --epochs 30

# High learning rate for faster convergence (watch for overfitting)
python cnn_training.py --model resnet50 --lr 1e-3 --epochs 50
```

**Process**:
1. Load images from `good/` and `bad/` directories
2. Split into train (70%), validation (15%), test (15%)
3. Apply data augmentation (crops, flips, color jitter)
4. Train with Adam optimizer + learning rate scheduling
5. Save best model based on validation accuracy
6. Generate evaluation plots and Grad-CAM visualizations

**Outputs**:
- `results/best_model.pth` - Best trained weights
- `results/training_config.json` - Hyperparameters and split info
- `results/training_curves.png` - Loss/accuracy vs epoch
- `results/evaluation_results.png` - Confusion matrix, ROC, PR curves
- `results/gradcam_results.png` - Attention maps on sample images

**Expected Training Time**: 
- ResNet-50: ~3-5 min (500 samples)
- EfficientNet-B0: ~2-4 min (optimized architecture)
- VGG-16: ~5-7 min (slower, heavier)

---

### Step 3: Run Inference

**Purpose**: Make predictions on new chart images with optional Grad-CAM visualizations.

```bash
python inference.py [OPTIONS]
```

**Options**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image` | str | `None` | Path to a single chart image |
| `--image_dir` | str | `None` | Path to folder of chart images |
| `--model_path` | str | `./results/best_model.pth` | Path to trained model weights |
| `--model_name` | choice | `resnet50` | Model architecture (must match training) |
| `--gradcam` | flag | `False` | Generate Grad-CAM heatmap visualizations |
| `--output_dir` | str | `./predictions` | Directory to save predictions and Grad-CAM images |

**Note**: Use either `--image` OR `--image_dir`, not both.

**Examples**:

```bash
# Single image prediction
python inference.py --image chart.png

# Batch process entire folder
python inference.py --image_dir ./test_charts/ --model_name resnet50

# Generate Grad-CAM visualizations
python inference.py --image chart.png --gradcam --output_dir ./explanations

# With custom model (EfficientNet)
python inference.py --image_dir ./charts/ --model_name efficientnet_b0 \
  --model_path ./results/best_model.pth --gradcam

# Batch processing with Grad-CAM
python inference.py --image_dir ./validation_set/ --model_name vgg16 --gradcam
```

**Output Format**:
- Console: Predictions displayed with label and confidence
- JSON: `predictions/results.json` containing all results
- Images: `predictions/gradcam_*.png` (if `--gradcam` flag used)

**Example Results.json**:
```json
{
  "predictions": [
    {
      "image": "chart1.png",
      "label": "GOOD",
      "probability": 0.89,
      "confidence": "High"
    },
    {
      "image": "chart2.png",
      "label": "BAD",
      "probability": 0.72,
      "confidence": "Medium"
    }
  ],
  "model_name": "resnet50",
  "timestamp": "2026-02-20T15:30:45.123456"
}
```

---

### Step 4: Launch Streamlit Web UI

**Purpose**: Interactive web interface for chart classification with real-time model selection.

```bash
streamlit run app.py
```

**Features**:
- 📤 **Image Upload**: Drag-and-drop or click to upload PNG/JPG/JPEG chart images
- 🤖 **Model Selection**: Dropdown menu auto-discovers all trained models from `./results/`
- 📊 **Prediction Display**: 
  - Classification result (GOOD / BAD)
  - Confidence level indicator
  - Probability score
  - Model metadata (training params, accuracy)
- 🔍 **Grad-CAM Visualization**: 3-panel display showing:
  1. Original uploaded chart
  2. Attention heatmap (where model focused)
  3. Overlay (heatmap on original)

**Usage**:

1. **Start Server**:
   ```bash
   streamlit run app.py
   ```

2. **Access Web Interface**:
   - Open browser to `http://localhost:8501`
   - UI loads automatically (may take 10-15 seconds first time due to model caching)

3. **Make a Prediction**:
   - Select model from dropdown
   - Upload chart image (PNG/JPG)
   - View results and Grad-CAM visualization
   - Upload another image to test different models

4. **Stop Server**:
   - Press `Ctrl+C` in terminal
   - Or close browser and server will auto-stop after idle timeout

**URL**: `http://localhost:8501`

**Performance**: 
- Model caching (loaded once, reused across sessions)
- Predictions typically <1 second per image
- Supports GPU acceleration if CUDA available

---

## 📊 Expected Results

### Dataset Statistics
```
Generated 1000 chart pairs:
├── good/ (500 charts)
├── bad/  (500 charts)
└── Split: 70% train (700), 15% val (150), 15% test (150)
```

### Model Performance (ResNet-50, 25 epochs, 500 samples)
```
Test Accuracy:  ~88-92%
Precision:      ~85-90%
Recall:         ~87-92%
ROC-AUC:        ~0.92-0.96
```

### Chart Quality Violations Detected
- **Chartjunk**: Excessive gridlines, decorative elements
- **3D Effects**: Misleading perspective distortions
- **Truncated Axes**: Non-zero baselines
- **Poor Color**: Rainbow palettes, low contrast
- **Data-Ink Ratio**: Redundant elements, heavy decoration
- **Dual Axes**: Misleading dual-axis combinations
- **Type Mismatch**: Pie charts >8 categories

---

## 🔧 Troubleshooting

### Issue: "CUDA out of memory" during training
```bash
# Reduce batch size
python cnn_training.py --batch_size 16

# Or use CPU explicitly
# The script auto-detects GPU, but uses CPU if not available
```

### Issue: Streamlit app won't start
```bash
# Reinstall streamlit
pip install --upgrade streamlit

# Check if port 8501 is in use
lsof -i :8501  # macOS/Linux

# Use alternative port
streamlit run app.py --server.port 8502
```

### Issue: Model not found when running inference
```bash
# Verify model path exists
ls -la ./results/best_model.pth

# Check training_config.json for model name
cat ./results/training_config.json | grep model_name

# Use correct model_name flag
python inference.py --image chart.png --model_name efficientnet_b0
```

### Issue: "No CUDA device detected"
```bash
# Check PyTorch setup
python -c "import torch; print(torch.cuda.is_available())"

# Falls back to CPU automatically (slower but functional)
```

### Issue: Synthetic data generation hangs
```bash
# Check available disk space
df -h

# Reduce number of samples
python synthetic_data_gen.py --num_samples 100

# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
```

---

## 🔬 Model Architecture Details

### ResNet-50 (Default)
```
ImageNet pretrained ResNet-50
+ Custom head: Dropout(0.3) → FC(2048→256) → ReLU → Dropout(0.2) → FC(256→1)
Transfer learning: Fine-tune last layer
```

### EfficientNet-B0
```
ImageNet pretrained EfficientNet-B0 (lighter weight)
+ Custom classifier: Dropout(0.3) → FC(1280→256) → ReLU → Dropout(0.2) → FC(256→1)
Best for edge deployment
```

### VGG-16
```
ImageNet pretrained VGG-16
+ Modified FC layers: Dropout(0.3) → FC(4096→256) → ReLU → Dropout(0.2) → FC(256→1)
Classical architecture, good interpretability
```

**Loss Function**: Binary Cross-Entropy with Sigmoid activation
**Optimizer**: Adam (lr=1e-4, betas=(0.9, 0.999))
**LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
**Early Stopping**: Patience=5 on validation accuracy

---

## 📦 Dependencies

See `requirements.txt` for complete list:

```
torch>=2.0.0              # PyTorch deep learning framework
torchvision>=0.15.0       # ImageNet models and transforms
matplotlib>=3.7.0         # Visualization and chart generation
Pillow>=9.0.0             # Image processing
numpy>=1.24.0             # Numerical computing
opencv-python>=4.7.0      # Computer vision utilities
scikit-learn>=1.3.0       # Metrics (confusion matrix, ROC, etc.)
tqdm>=4.65.0              # Progress bars
streamlit>=1.28.0         # Web UI framework
```

**Installation**:
```bash
pip install -r requirements.txt
```

---

## 🎯 Common Workflows

### A. Train and Deploy on Custom Dataset
```bash
# 1. Prepare data in good/ and bad/ directories
# 2. Train model
python cnn_training.py --data_dir ./my_dataset --epochs 30

# 3. Verify model saved
ls results/

# 4. Launch UI (auto-detects new model)
streamlit run app.py
```

### B. Compare Multiple Model Architectures
```bash
# Train ResNet-50
python cnn_training.py --model resnet50 --output_dir ./results_resnet

# Train EfficientNet-B0
python cnn_training.py --model efficientnet_b0 --output_dir ./results_efficient

# Train VGG-16
python cnn_training.py --model vgg16 --output_dir ./results_vgg

# Compare results in separate directories
```

### C. Exploratory Inference with Explanations
```bash
# Generate Grad-CAM for understanding model decisions
python inference.py --image_dir ./test_charts/ \
  --model_name resnet50 \
  --model_path ./results/best_model.pth \
  --gradcam \
  --output_dir ./gradcam_analysis
```

### D. Production Batch Scoring
```bash
# Score entire validation set, save JSON results
python inference.py --image_dir ./validation/ \
  --model_path ./results/best_model.pth \
  --output_dir ./validation_predictions

# Results in validation_predictions/results.json
cat ./validation_predictions/results.json | jq '.predictions[] | {image, label, probability}'
```

---

## 📝 Citation & References

**Tufte's Principles** (Foundations):
- Edward R. Tufte, "The Visual Display of Quantitative Information" (2nd ed., 2001)
- Maximize data-ink ratio
- Avoid chartjunk and decorative elements
- Use appropriate chart types
- Ensure clear, high-contrast design

**Deep Learning Architectures**:
- He et al., "Deep Residual Learning for Image Recognition" (ResNet, 2015)
- Tan & Le, "EfficientNet: Rethinking Model Scaling" (EfficientNet, 2019)
- Simonyan & Zisserman, "Very Deep Convolutional Networks" (VGG, 2014)

**Interpretability**:
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (2016)

---

## 📄 License

See [LICENSE](LICENSE) file.

---

## 🤝 Contributing

Suggestions for improvements:
- [ ] Add more chart types (waterfall, box plots, heatmaps)
- [ ] Extended violation types (e.g., accessibility, color-blind friendliness)
- [ ] Mobile-responsive Streamlit UI
- [ ] Real-world dataset integration
- [ ] API endpoint for model serving (Flask/FastAPI)

---

## 📞 Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Verify all dependencies installed: `pip list | grep -E "torch|streamlit"`
3. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
4. Review generated files exist in expected directories

---

**Last Updated**: February 20, 2026  
**Project Version**: 1.0  
**Python Version**: 3.11+
