# VisScore — synthetic Tufte / non-Tufte chart images

Python package for generating labeled PNGs (Tufte-style vs chartjunk) with **Hugging Face Diffusers** and open weights. No paid image APIs. Default batch: **150** + **150** images.

Layout follows a standard **`src/` package** layout so you can install with `pip`, run as a module or console script, and later deploy the same entrypoint on **Google Cloud** (Compute Engine GPU VM, Vertex AI custom job, GKE with NVIDIA device plugin, or Artifact Registry + batch worker) without notebooks.

---

## Install

From the repository root (creates console script **`visscore-generate`** and importable **`visscore_synthetic`**):

```bash
python -m pip install --upgrade pip setuptools wheel   # recommended; avoids old pip editable issues
pip install -e .
# equivalent:
pip install -r requirements.txt
```

Requires **Python 3.9+** and a **CUDA GPU** for practical runtimes (CPU is supported but very slow).

---

## Hugging Face token

**FLUX.1-schnell** is [gated](https://huggingface.co/black-forest-labs/FLUX.1-schnell): create a free account, accept the model terms, create a token, then either:

```bash
export HF_TOKEN=hf_...
```

or pass `--hf-token hf_...` on the CLI. **SDXL** may also require accepting terms on Hugging Face.

---

## Run (CLI)

After `pip install -e .`:

```bash
visscore-generate \
  --out-dir data/synthetic \
  --n-tufte 150 \
  --n-non-tufte 150 \
  --model flux-schnell \
  --image-size 768 \
  --seed 42
```

Equivalent without installing a script (if `PYTHONPATH=src` or after editable install):

```bash
python -m visscore_synthetic \
  --out-dir data/synthetic \
  --n-tufte 150 \
  --n-non-tufte 150
```

**CUDA OOM:** use `--model sdxl --image-size 512` and free GPU memory (restart runtime if needed).

---

## Google Colab (GPU, no notebook required)

Use a **terminal** in Colab (or `%pip` / `!` in a one-off cell only to run shell—your pipeline stays CLI-only).

```bash
git clone -b image-generation --single-branch https://github.com/YOUR_USER/VisScore.git /content/VisScore
cd /content/VisScore
python -m pip install --upgrade pip setuptools wheel
pip install -e .
export HF_TOKEN=hf_your_token_here
python -m visscore_synthetic \
  --out-dir /content/visscore_synthetic \
  --n-tufte 150 \
  --n-non-tufte 150 \
  --model flux-schnell \
  --image-size 768
```

`/content` is ephemeral; write to **Google Drive** or **Cloud Storage** (e.g. mount a bucket with gcsfuse, or upload after `tar`) if you need to keep outputs.

---

## Google Cloud (later scale-out)

Same code path as anywhere else: install the package (or use the container) and invoke **`visscore-generate`** or **`python -m visscore_synthetic`**.

- **Compute Engine:** GPU VM (L4 / T4 / A100 per quota), Docker or raw venv, set `HF_TOKEN` from Secret Manager or metadata.
- **Vertex AI:** Custom training or custom container job; pass secrets as environment variables; write `--out-dir` to a mounted Cloud Storage path (e.g. job output URI).
- **Artifact Registry:** Build and push the image from the included [`Dockerfile`](Dockerfile) (`docker build -t visscore-synthetic .`), then run that image on a GPU node with `HF_TOKEN` injected.

The repository intentionally does **not** depend on `.ipynb` files so jobs are reproducible from versioned CLI + container.

---

## CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--out-dir` | `data/synthetic` | Output root; creates `tufte/` and `non_tufte/` |
| `--n-tufte` | `150` | Tufte-style image count |
| `--n-non-tufte` | `150` | Non-Tufte image count |
| `--model` | `flux-schnell` | `flux-schnell` or `sdxl` |
| `--image-size` | `768` | Square side in pixels (multiple of 8) |
| `--seed` | `42` | Base random seed |
| `--hf-token` | env | Overrides `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` |
| `--flux-model-id` | BFL schnell id | Override checkpoint |
| `--sdxl-model-id` | SDXL base id | Override checkpoint |

---

## Output layout

```
<out-dir>/
├── tufte/tufte_0000.png …
└── non_tufte/non_tufte_0000.png …
```

`data/synthetic/` is in `.gitignore`.

---

## Repository layout

```
VisScore/
├── src/
│   └── visscore_synthetic/
│       ├── __init__.py
│       ├── __main__.py      # python -m visscore_synthetic
│       ├── cli.py           # argparse + batch loop
│       ├── pipelines.py     # Diffusers load / generate
│       └── prompts.py       # prompt lists
├── pyproject.toml           # package metadata + visscore-generate entry point
├── setup.py                 # setuptools shim for older pip editable installs
├── requirements.txt         # pip install -e .
├── Dockerfile               # optional GPU container for GCP / GCE
└── README.md
```

---

## Notes

- Diffusion outputs are **stylized** charts; numeric accuracy is not guaranteed—suitable for **rough** binary aesthetic classification.
- Respect each model’s **license** on Hugging Face (e.g. FLUX.1-schnell Apache-2.0; verify SDXL terms if used).
