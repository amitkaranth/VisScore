# VisScore: CNN-Based Visualization Quality Classifier

A machine learning system for assessing visualization quality based on Edward Tufte’s design principles. It supports **synthetic training data**, **charts generated from your own CSVs**, and a **combined CNN + vision–language model (VLM)** workflow for informed good/bad decisions with optional reasoning text.

## 📋 Project Overview

**Problem**: Automatically assess whether a data visualization follows best practices (Tufte-aligned “good” vs problematic “bad”).

**Solution**:
- **Train** a CNN (ResNet-50, EfficientNet-B0, or VGG-16) on synthetic good/bad chart images—or fine-tune with your own labeled exports.
- **Generate** Tufte-style charts (and optional “bad” counterparts) from arbitrary CSVs for exploration or extra training data.
- **Score** any chart PNG with the **retrained CNN**, optionally plus a **VLM** (Gemini or Groq) and a **consensus** rule in Streamlit or the CLI.

**Key Features**:
- Synthetic data generation (Matplotlib and optional Plotly/Kaleido)
- **CSV → charts**: `csv_tufte_charts.py` (good + bad variants; metric columns preferred over IDs; single-hue good bars)
- **Standalone Streamlit** for CSV uploads: `streamlit_csv_charts.py` (separate from the main classifier UI)
- CNN training, batch inference, Grad-CAM
- **Multimodal**: CNN + VLM Tufte-style judge + consensus (`vlm_judge.py`, `multimodal_inference.py`, sidebar in `app.py`)
- Main Streamlit UI: model pickers, VLM provider/keys, combined decision

---

## 🧭 Recommended end-to-end flow

Use this narrative when you want **real datasets → plots → informed scoring**:

1. **Generate plots from any CSV dataset**  
   - **CLI:** `python csv_tufte_charts.py --input_dir ./your_csvs --output_dir ./generated_plots`  
   - **UI:** `streamlit run streamlit_csv_charts.py` — upload CSVs, set DPI / max chart types / include bad variants, download PNGs + manifest.  
   The tool infers dates, categories, and numeric columns, prefers **real metrics** (rates, deaths, burden, etc.) over **ID-like** columns (ISO codes, country codes), and draws **good** bars in a **single color** (better alignment with CNN training).

2. **(Parallel track) Train or retrain the CNN**  
   Use synthetic data, your own `good/` and `bad/` folders, or mix in exports from step 1 if you label them.  
   ```bash
   python cnn_training.py --data_dir ./visual_dataset --model resnet50 --epochs 25
   ```  
   Weights land in `./results/` (e.g. `best_model.pth` + `training_config.json`).

3. **Link charts to the inference service**  
   - **Main app:** `streamlit run app.py` — upload a generated PNG (or any chart), pick your **retrained** checkpoint from the sidebar, optionally enable **VLM + consensus** and set **Gemini** or **Groq** keys (see `.env.example`).  
   - **CLI:** `python multimodal_inference.py --image ./generated_plots/.../chart.png --vlm_provider groq` (or default Gemini).  
   You get **CNN probabilities**, optional **VLM verdict + reasoning**, and a **combined label** (agreement, SPLIT, or CNN-only fallback).

This keeps **plot generation** (`csv_tufte_charts` / `streamlit_csv_charts`) independent from **scoring** (`app.py` / `inference.py` / `multimodal_inference.py`), while the workflow is easy to chain: export PNGs → open them in VisScore.

**Full “from scratch” guide (data folders, all training flags, VLMs, both Streamlit apps):** see **[FROM_SCRATCH.md](FROM_SCRATCH.md)**. **One-shot pipeline** (install + synthetic data + train): `./run_from_scratch.sh all` (see script header for `install` / `data` / `train` subcommands).

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Navigate to project directory
cd /path/to/VisScore

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

### 6. Multimodal inference (retrained CNN + VLM, combined decision)

Combine your **trained CNN** with a **vision–language model** for Tufte-oriented **text reasoning** plus a **consensus** label.

**Providers**
- **Gemini:** key from [Google AI Studio](https://aistudio.google.com/apikey); `pip install google-generativeai` (in `requirements.txt`).
- **Groq:** vision-capable chat models; set `GROQ_API_KEY` and use `--vlm_provider groq` (CLI) or choose Groq in the Streamlit sidebar.

**Environment (optional):** copy `.env.example` to `.env` and set `GEMINI_API_KEY`, `GROQ_API_KEY`, `VLM_PROVIDER`, etc.

**CLI** (writes `predictions_multimodal/multimodal_results.json`):

```bash
export GEMINI_API_KEY="your_key"
python multimodal_inference.py --image ./generated_plots/my_run/chart.png --model_name resnet50 --gradcam

# Groq example
export GROQ_API_KEY="your_key"
python multimodal_inference.py --image ./chart.png --vlm_provider groq --model_name resnet50
```

**Streamlit (`app.py`):** enable **“Enable VLM + consensus”**, choose provider, paste keys / model id as needed.

**Consensus (high level):** when CNN and VLM **agree**, that label wins; when they **disagree**, the UI shows **SPLIT** so you can compare logits and VLM reasoning; on API failure, results fall back to **CNN only**. The VLM uses a **strict Tufte-style rubric** (data-ink, grid weight, chartjunk, on-image captions, etc.).

Gemini calls try **fallback model IDs** if one returns 404 or quota errors.

### 7. CSV → charts (optional, any dataset)

**CLI — batch over a folder or glob:**

```bash
python csv_tufte_charts.py --input_dir ./raw_csvs --output_dir ./generated_plots
python csv_tufte_charts.py --input_glob "./data/*.csv" --output_dir ./out --dpi 200 --no_bad
```

- Writes one subfolder per CSV stem, **good** Tufte-style PNGs per inferred chart type (line, bar, scatter, histogram), and by default matching **bad** (chartjunk) variants.
- Outputs `csv_tufte_manifest.json` under the output directory.

**Streamlit — upload CSVs only (separate app):**

```bash
streamlit run streamlit_csv_charts.py
```

Does **not** modify `app.py`. Use the generated PNGs as input to `app.py` or `multimodal_inference.py`.

---

## 📁 Project Structure

```
VisScore/
├── synthetic_data_gen.py       # Generate synthetic chart dataset (matplotlib)
├── synthetic_data_gen_plotly.py  # Optional Plotly/Kaleido charts
├── csv_tufte_charts.py         # CSV → good (+ optional bad) chart PNGs (standalone)
├── streamlit_csv_charts.py     # Streamlit UI for CSV uploads → charts (standalone)
├── cnn_training.py             # Train CNN models
├── inference.py                # Batch inference + Grad-CAM
├── multimodal_inference.py     # CNN + VLM + consensus (CLI)
├── vlm_judge.py                # VLM prompts (Gemini/Groq), JSON parse, consensus helpers
├── app.py                      # Main Streamlit: CNN + optional VLM + Grad-CAM
├── .env.example                # Example API keys / VLM_PROVIDER (copy to .env)
├── FROM_SCRATCH.md             # End-to-end: data layout, training options, VLMs, Streamlit
├── run_from_scratch.sh         # Optional: pip install + synthetic data + train
├── requirements.txt
├── README.md
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
├── generated_plots/         # Typical output from csv_tufte_charts.py (you create this)
│   └── csv_tufte_manifest.json
│
└── __pycache__/             # Python cache (ignore)
```

---

## 📈 CSV → charts (`csv_tufte_charts.py`) — details

**Purpose:** Turn arbitrary CSV files into static chart images for review, reporting, or feeding the VisScore CNN/VLM.

**Behavior:**
- Infers **datetime**, **numeric**, and **categorical** columns; builds line, bar (horizontal or vertical), scatter, and histogram when data allows.
- **Good** charts: white background, minimal grid, single-hue categorical bars (`#4C72B0`), honest bar baseline.
- **Bad** charts (default): chartjunk-style twins (heavy grids, rainbow fills, truncated bars, etc.) for contrast or extra training negatives.
- **Numeric column choice:** columns whose names look like **metrics** (burden, deaths, incidence, rate, `100k`, etc.) are preferred over **IDs** (ISO, country code, numeric territory codes, postal, etc.).

| Argument | Description |
|----------|-------------|
| `--input_dir` | Folder of `.csv` files |
| `--input_glob` | e.g. `./exports/*.csv` |
| `--output_dir` | Root for PNGs + `csv_tufte_manifest.json` |
| `--dpi` | Raster resolution (default 150) |
| `--max_charts_per_file` | Cap on chart *types* per CSV (each type may emit good + bad) |
| `--no_bad` | Only write Tufte-style charts |

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

**Example `results.json` fields** (per image): `label` (GOOD/BAD), `probability` (P(good) after sigmoid), `confidence` (model certainty in the displayed label).

---

### Step 4: Launch Streamlit Web UI

**Purpose**: Interactive web interface for chart classification with real-time model selection.

```bash
streamlit run app.py
```

**Features**:
- **Image upload**: PNG/JPEG chart images (including exports from `csv_tufte_charts.py`)
- **Model selection**: Dropdown discovers checkpoints in `./results/` (active `training_config.json` + `best_model.pth`, plus tagged pairs like `training_config_<tag>.json` / `best_model_<tag>.pth`)
- **CNN output**: label (GOOD/BAD), **probability** ≈ P(good), **confidence** in the chosen label, optional Grad-CAM
- **Multimodal**: optional VLM (Gemini or Groq) + **consensus** with CNN (agree / SPLIT / CNN-only fallback)
- **Grad-CAM**: original, heatmap, overlay

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

### E. Real data → charts → CNN + VLM decision
```bash
# 1) Generate plots from CSVs
python csv_tufte_charts.py --input_dir ./datasets --output_dir ./generated_plots

# 2) Score one chart with CNN + VLM (Groq example)
export GROQ_API_KEY="..."
python multimodal_inference.py \
  --image ./generated_plots/some_stem/some_stem_bar_....png \
  --model_path ./results/best_model.pth \
  --model_name resnet50 \
  --vlm_provider groq

# Or use the main UI: streamlit run app.py → upload the same PNG, enable VLM + consensus
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
- [ ] Add more chart types in `csv_tufte_charts.py` (e.g. box plots, small multiples)
- [ ] Extended violation types (accessibility, color-blind checks)
- [ ] API endpoint for model serving (Flask/FastAPI)

---

## 📞 Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Verify all dependencies installed: `pip list | grep -E "torch|streamlit"`
3. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
4. Review generated files exist in expected directories

---

**Last Updated**: April 6, 2026  
**Python Version**: 3.11+
