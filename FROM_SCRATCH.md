# VisScore: run everything from scratch

This guide is separate from the short [README.md](README.md). It explains **where data lives**, **how to train**, **which VLMs are used and what they do**, and **how to run every Streamlit app and CLI entrypoint**.

---

## 1. Big picture

| Stage | What | Main command(s) |
|--------|------|-------------------|
| A. Training data | Synthetic good/bad PNGs (or your own folders) | `synthetic_data_gen.py` |
| B. Train CNN | Learn good vs bad from images | `cnn_training.py` |
| C. (Optional) Real CSV charts | Tufte-style + bad twins from CSVs | `csv_tufte_charts.py` or `streamlit_csv_charts.py` |
| D. Score charts | CNN only or CNN + VLM consensus | `inference.py`, `multimodal_inference.py`, `streamlit run app.py` |

**Automated pipeline (synth data + train):** from the repo root, after editing variables at the top of the script:

```bash
chmod +x run_from_scratch.sh
./run_from_scratch.sh all        # install deps, generate data, train
./run_from_scratch.sh install    # only pip install (-r requirements.txt)
./run_from_scratch.sh data       # only synthetic dataset
./run_from_scratch.sh train      # only training (expects data_dir to exist)
```

Override paths without editing the file:

```bash
DATA_DIR=./my_dataset NUM_SAMPLES=1000 EPOCHS=30 ./run_from_scratch.sh all
```

---

## 2. Where to put data

### 2.1 For **CNN training** (`cnn_training.py`)

Your dataset directory must look like this:

```text
your_dataset/
├── good/          # Tufte-style / acceptable charts (PNG/JPG)
│   ├── a.png
│   └── ...
└── bad/           # Problematic charts
    ├── b.png
    └── ...
```

- **Default** in docs and scripts: `./vis_dataset` (created by `synthetic_data_gen.py`).
- **You can use any path:** pass `--data_dir /path/to/your_dataset`.
- Labels are implicit: everything under `good/` is class **good (1)**, everything under `bad/` is **bad (0)**.

### 2.2 **Synthetic** data (no manual labeling)

`python synthetic_data_gen.py` writes:

```text
vis_dataset/   # or whatever you pass as --output_dir
├── good/
├── bad/
└── metadata.json
```

### 2.3 **CSV → images** (optional, not required for training)

Put CSV files anywhere; point the tool at a folder or glob:

```bash
python csv_tufte_charts.py --input_dir ./raw_csvs --output_dir ./generated_plots
```

Output layout:

```text
generated_plots/
├── <csv_stem>/
│   ├── *.png
│   └── ...
└── csv_tufte_manifest.json
```

To **train** on those PNGs, you must **copy or symlink** them into `good/` and `bad/` yourself (or extend your pipeline). The CSV tool is independent of the trainer.

### 2.4 **Trained model** output

After training:

```text
results/                    # default --output_dir
├── best_model.pth
├── training_config.json
├── training_curves.png
├── evaluation_results.png
└── gradcam_results.png
```

Point inference and Streamlit at `./results/best_model.pth` (and matching `--model_name`).

---

## 3. Training: options reference

### 3.1 `synthetic_data_gen.py`

| Flag | Default | Meaning |
|------|---------|---------|
| `--output_dir` | `./vis_dataset` | Where `good/`, `bad/`, `metadata.json` go |
| `--num_samples` | `500` | Number of **pairs** (good + bad each) |
| `--seed` | `42` | RNG seed |
| `--augment_strength` | `medium` | `none` / `low` / `medium` / `high` (post-save image aug) |
| `--seed_aug` | (none) | Optional seed for augmentations |

Example:

```bash
python synthetic_data_gen.py --output_dir ./vis_dataset --num_samples 800 --augment_strength medium
```

Optional Plotly/Kaleido generator (separate script): `synthetic_data_gen_plotly.py` (see that file’s `--help`).

### 3.2 `cnn_training.py`

| Flag | Default | Meaning |
|------|---------|---------|
| `--data_dir` | `./vis_dataset` | Folder with `good/` and `bad/` |
| `--output_dir` | `./results` | Checkpoints and plots |
| `--model` | `resnet50` | `resnet50`, `efficientnet_b0`, or `vgg16` |
| `--epochs` | `25` | Training epochs |
| `--batch_size` | `32` | Batch size (lower if OOM) |
| `--lr` | `1e-4` | Adam learning rate |
| `--img_size` | `224` | Input size |
| `--val_split` | `0.15` | Validation fraction |
| `--test_split` | `0.15` | Test fraction |
| `--seed` | `42` | Reproducibility |

Example:

```bash
python cnn_training.py --data_dir ./vis_dataset --output_dir ./results --model resnet50 --epochs 25 --batch_size 32
```

**Important:** `--model` at inference time must match the architecture used when the weights were saved.

---

## 4. Vision–language models (VLMs): what we use and what happens

VisScore can combine:

1. **Your CNN** — outputs a logit → **sigmoid** → **probability of “good”**; label GOOD/BAD using the app’s threshold; optional **Grad-CAM**.
2. **A VLM** — sends the **same image** plus a **strict Tufte-style text rubric**; the model returns JSON: `GOOD` or `BAD` and short **reasoning**.

Code lives mainly in **`vlm_judge.py`** (prompts, Gemini/Groq clients, parsing) and **`multimodal_inference.py`** / **`app.py`** (wiring + consensus).

### 4.1 Providers

| Provider | Typical use | API key env var | Notes |
|----------|-------------|-----------------|--------|
| **Google Gemini** | Default in many setups | `GEMINI_API_KEY` | Free tier via [Google AI Studio](https://aistudio.google.com/apikey). Client: `google-generativeai`. The code may try several model IDs if one 404s or hits quota. |
| **Groq** | Fast vision chat | `GROQ_API_KEY` | [Groq console](https://console.groq.com/keys). Set `VLM_PROVIDER=groq` or pass `--vlm_provider groq`. Default vision model id is configured for Llama 4 Scout–class IDs (see `.env.example` / Streamlit sidebar). |

Copy **`.env.example`** to **`.env`** and fill keys, or `export` variables in your shell. Never commit real keys.

### 4.2 What “consensus” means

- If **CNN** and **VLM** **agree** on GOOD or BAD → that label is the combined result.
- If they **disagree** → the UI/CLI reports something like **SPLIT** (two voters, no majority rule beyond “flag for human review”).
- If the VLM **fails** (network, key, rate limit) → fallback to **CNN only**.

The VLM does **not** retrain the CNN; it is an **second opinion** with natural-language justification.

---

## 5. How to run: CLIs

### 5.1 CNN-only inference

```bash
python inference.py --image ./chart.png --model_path ./results/best_model.pth --model_name resnet50
python inference.py --image_dir ./folder --model_path ./results/best_model.pth --gradcam --output_dir ./predictions
```

### 5.2 CNN + VLM (multimodal)

```bash
export GEMINI_API_KEY="..."
python multimodal_inference.py --image ./chart.png --model_name resnet50 --model_path ./results/best_model.pth

export GROQ_API_KEY="..."
export VLM_PROVIDER=groq
python multimodal_inference.py --image ./chart.png --vlm_provider groq --model_name resnet50
```

Outputs default to `./predictions_multimodal/` (JSON + optional Grad-CAM).

### 5.3 CSV → charts

```bash
python csv_tufte_charts.py --input_dir ./csvs --output_dir ./generated_plots
python csv_tufte_charts.py --input_glob "./data/*.csv" --output_dir ./out --no_bad
```

---

## 6. How to run: Streamlit apps

Two **separate** apps (different ports if you run both).

| App | Purpose | Command | Default URL |
|-----|---------|---------|-------------|
| **Main VisScore** | Upload chart → CNN prediction, Grad-CAM, optional VLM + consensus | `streamlit run app.py` | http://localhost:8501 |
| **CSV charts only** | Upload CSVs → generate good/bad PNGs + download ZIP | `streamlit run streamlit_csv_charts.py` | http://localhost:8501 (stop the other first, or use another port) |

**Second app on another port:**

```bash
streamlit run streamlit_csv_charts.py --server.port 8502
```

---

## 7. Suggested first-time sequence

1. `python3 -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. `./run_from_scratch.sh all` **or** manually: `synthetic_data_gen.py` then `cnn_training.py`
4. `streamlit run app.py` — upload a test image; optionally enable VLM and add keys in the sidebar
5. (Optional) `streamlit run streamlit_csv_charts.py` — generate plots from CSVs, then upload those PNGs in `app.py`

For the full narrative (CSV → plots → score with retrained CNN + VLM), see **Recommended end-to-end flow** in [README.md](README.md).
