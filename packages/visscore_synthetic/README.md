# VisScore — programmatic synthetic chart dataset (Tufte vs non-Tufte)

Generates **labeled PNG charts** for CNN training using **matplotlib** and **seaborn** (no diffusion, no Hugging Face, no GPU). Two classes:

- `tufte/` — high data-ink, minimal decoration, restrained palettes, thin axes, 2D statistical graphics  
- `non_tufte/` — chartjunk: loud colors, heavy grids, legends, 3D bars, busy dashboards, BI-style clutter  

Use this alongside **your own** sources (e.g. **Plotly** exports, **Reddit** scrapes) so the model sees more than one renderer.

**Outputs**

- `data/synthetic/tufte/` and `data/synthetic/non_tufte/` (or `--out-dir`)  
- `metadata.jsonl` — one JSON object per image (`renderer`, `chart_type`, sizes, seeds, …) unless `--no-metadata`

---

## Dependencies

From `pyproject.toml`: **matplotlib**, **numpy**, **Pillow**, **pandas**, **seaborn**.

---

## Quick start (copy-paste)

This package lives under **`packages/visscore_synthetic/`** in the VisScore monorepo. From the **repository root**:

```bash
cd /path/to/VisScore
```

```bash
python -m pip install --upgrade pip setuptools wheel
```

```bash
pip install -e ./packages/visscore_synthetic
```

**Small run** (20 + 20 images, square 512px, default metadata):

```bash
visscore-generate --n-tufte 20 --n-non-tufte 20 --seed 0 --out-dir data/synthetic --dpi 100 --image-size 512
```

If `visscore-generate` is not on `PATH`:

```bash
python -m visscore_synthetic --n-tufte 20 --n-non-tufte 20 --seed 0 --out-dir data/synthetic --dpi 100 --image-size 512
```

**Default counts** (150 Tufte + 150 non-Tufte, matplotlib + seaborn):

```bash
visscore-generate --out-dir data/synthetic --seed 42
```

**Matplotlib only** (faster; skips all `sns_*` chart types):

```bash
visscore-generate --libraries matplotlib --n-tufte 50 --n-non-tufte 50 --out-dir data/synthetic
```

**Seaborn only:**

```bash
visscore-generate --libraries seaborn --n-tufte 30 --n-non-tufte 30 --out-dir data/synthetic
```

---

## More examples

Augmentation + custom metadata path:

```bash
visscore-generate --n-tufte 50 --n-non-tufte 50 --seed 1 --augment --style-strength 0.3 \
  --metadata ./runs/run1/metadata.jsonl
```

Heavier matplotlib “skin” variety on **matplotlib** charts (`extended` adds dark/grid styles for `non_tufte` only):

```bash
visscore-generate --matplotlib-style-mode extended --n-tufte 30 --n-non-tufte 30 --out-dir data/synthetic
```

Exclude a **slow** seaborn type (`sns_clustermap_busy` can take much longer per image on CPU):

```bash
visscore-generate --non-tufte-charts sns_heatmap_loud,sns_box_swarm_busy,sns_kde_overlay_loud,sns_bar_estimator_show,sns_scatter_hue_size \
  --n-tufte 100 --n-non-tufte 100 --out-dir data/synthetic
```

---

## CLI reference

| Option | Default | Description |
|--------|---------|-------------|
| `--out-dir` | `data/synthetic` | Root; creates `tufte/` and `non_tufte/` |
| `--n-tufte` | 150 | Tufte-class image count |
| `--n-non-tufte` | 150 | Non-Tufte image count |
| `--seed` | 42 | Global seed (reproducible dataset) |
| `--dpi` | 100 | Figure DPI when saving |
| `--min-width` / `--max-width` | 480 / 896 | Random width per image (px) |
| `--min-height` / `--max-height` | 360 / 672 | Random height per image (px) |
| `--image-size` | — | Square size (overrides min/max width & height) |
| `--libraries` | `matplotlib,seaborn` | Comma list: `matplotlib`, `seaborn` |
| `--matplotlib-style-mode` | `light` | `none` \| `light` \| `extended` — random `plt.style` for **matplotlib** charts only; `extended` adds dark/grid styles for `non_tufte` |
| `--tufte-charts` | all | Comma-separated subset of Tufte keys |
| `--non-tufte-charts` | all | Comma-separated subset of non-Tufte keys |
| `--augment` | off | PIL: blur / JPEG / brightness / contrast |
| `--style-strength` | 0.35 | Augmentation strength in [0, 1] |
| `--metadata` | `<out-dir>/metadata.jsonl` | JSONL path |
| `--no-metadata` | — | Skip metadata file |

### Chart keys (matplotlib)

**Tufte:** `line`, `scatter`, `bar_horizontal`, `dot_strip`, `small_multiples`, `box`, `sparkline`, `area`

**Non-Tufte:** `bar_rainbow`, `line_clutter`, `pie_exploded`, `dashboard`, `bar3d`, `scatter_annotated`, `histogram_busy`

### Chart keys (seaborn)

**Tufte:** `sns_reg_minimal`, `sns_kde_1d`, `sns_violin_light`, `sns_heatmap_muted`, `sns_line_facet_subtle`

**Non-Tufte:** `sns_heatmap_loud`, `sns_box_swarm_busy`, `sns_kde_overlay_loud`, `sns_bar_estimator_show`, `sns_scatter_hue_size`, `sns_clustermap_busy`

---

## Filenames

`tufte_<tag>_s<seed>_i<index>.png` and `non_tufte_<tag>_s<seed>_i<index>.png` — `<tag>` is `mpl` (matplotlib) or `sns` (seaborn). JSONL metadata still has full `renderer`: `matplotlib` / `seaborn`. Deterministic from `--seed` and index.

---

## Package layout

```
src/visscore_synthetic/
  __init__.py
  __main__.py           # python -m visscore_synthetic
  cli.py                # CLI entrypoint
  seeding.py            # deterministic RNG per image
  registry.py           # merge matplotlib + seaborn; dispatcher
  mpl_styles.py         # matplotlib style pools for mpl charts
  tufte_charts.py       # matplotlib Tufte-style
  non_tufte_charts.py   # matplotlib chartjunk
  sns_tufte_charts.py   # seaborn Tufte-style
  sns_non_tufte_charts.py
  pipelines.py          # figure → PNG (+ optional augment)
  augment.py
  metadata.py
```

---

## Docker

From the **repository root** (build context must be this package directory):

```bash
docker build -t visscore-synthetic -f packages/visscore_synthetic/Dockerfile packages/visscore_synthetic
docker run --rm -v "$PWD/out:/out" visscore-synthetic --out-dir /out --n-tufte 10 --n-non-tufte 10
```

Or `cd packages/visscore_synthetic` and run `docker build -t visscore-synthetic .` as before.

---

## Notes

- Charts are **programmatic** (matplotlib/seaborn), not text-to-image diffusion.  
- Metadata includes `renderer` (`matplotlib` or `seaborn`) and sometimes `matplotlib_style` when a style context was applied.  
- Same `--seed` and counts reproduce the same images for fixed package versions.  
- For large batches on CPU, prefer **`--libraries matplotlib`** or omit **`sns_clustermap_busy`** from `--non-tufte-charts` if runs are too slow.
