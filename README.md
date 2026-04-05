# VisScore — synthetic Tufte / non-Tufte chart images

Generate labeled PNGs for CNN training using **open** text-to-image weights (Hugging Face Diffusers). No paid image APIs. Default: **150** Tufte-style + **150** chartjunk-style images.

**Models**

- **FLUX.1-schnell** (default): best quality/speed tradeoff on Colab; [gated on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-schnell) (free account + accept terms + `HF_TOKEN`).
- **SDXL** (fallback): lower VRAM; pass `--model sdxl` if you hit CUDA OOM.

---

## Google Colab (GPU)

1. **Runtime → Change runtime type →** select a **GPU** (e.g. T4).
2. Clone **this branch** (replace the URL with your fork or upstream):

```bash
git clone -b image-generation --single-branch https://github.com/YOUR_USER/VisScore.git /content/VisScore
cd /content/VisScore
```

3. Install Python deps (Colab already ships `torch` with CUDA in most runtimes):

```bash
pip install -r requirements.txt
```

4. Authenticate with Hugging Face (needed for gated FLUX weights). Either:

- Set a Colab secret named **`HF_TOKEN`**, then in a notebook cell:

```python
import os
from google.colab import userdata
from huggingface_hub import login
token = userdata.get("HF_TOKEN")
login(token=token)
os.environ["HF_TOKEN"] = token
```

- Or run `huggingface-cli login` / `python -c "from huggingface_hub import login; login()"` and follow the prompt.

5. Run the generator (writes under `/content/out` on the VM):

```bash
export HF_TOKEN=hf_your_token_here   # if not already set by login()
python scripts/generate_tufte_synthetic.py \
  --out-dir /content/visscore_synthetic \
  --n-tufte 150 \
  --n-non-tufte 150 \
  --model flux-schnell \
  --image-size 768 \
  --seed 42
```

6. **Optional:** open [notebooks/colab_generate_tufte_synthetic.ipynb](notebooks/colab_generate_tufte_synthetic.ipynb) in Colab and run cells in order (same effect as the commands above).

7. **Download outputs:** Colab discards `/content` when the session ends. Zip and download, or set `--out-dir` to a mounted Google Drive path, e.g. `/content/drive/MyDrive/visscore_synthetic` after `drive.mount("/content/drive")`.

---

## Local CLI (CUDA machine)

From the repo root:

```bash
cd /path/to/VisScore
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export HF_TOKEN=hf_...      # Linux/macOS; Windows: set HF_TOKEN=hf_...
python scripts/generate_tufte_synthetic.py --out-dir data/synthetic
```

---

## CLI reference

All arguments for [scripts/generate_tufte_synthetic.py](scripts/generate_tufte_synthetic.py):

| Argument | Default | Description |
|----------|---------|-------------|
| `--out-dir` | `data/synthetic` | Output root; creates `tufte/` and `non_tufte/` |
| `--n-tufte` | `150` | Count of Tufte-style images |
| `--n-non-tufte` | `150` | Count of non-Tufte images |
| `--model` | `flux-schnell` | `flux-schnell` or `sdxl` |
| `--image-size` | `768` | Square side in pixels (multiple of 8) |
| `--seed` | `42` | Base random seed |
| `--hf-token` | env | Overrides `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` |
| `--flux-model-id` | BFL schnell id | Override checkpoint |
| `--sdxl-model-id` | SDXL base id | Override checkpoint |

**Environment:** `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` for downloads of gated models.

**OOM:** Restart runtime, then e.g.  
`python scripts/generate_tufte_synthetic.py --out-dir /content/out --model sdxl --image-size 512`

---

## Output layout

```
<out-dir>/
├── tufte/tufte_0000.png …
└── non_tufte/non_tufte_0000.png …
```

`data/synthetic/` is listed in `.gitignore` so generated PNGs are not committed by default.

---

## Repository layout

```
VisScore/
├── scripts/
│   └── generate_tufte_synthetic.py
├── notebooks/
│   └── colab_generate_tufte_synthetic.ipynb
├── requirements.txt
└── README.md
```

---

## Notes

- Diffusion outputs are **stylized** charts; numeric accuracy and axis text are not guaranteed. Suitable for **rough** binary aesthetic classification.
- Respect each model’s **license** on Hugging Face (FLUX.1-schnell is Apache-2.0; check SDXL terms if you use it).
