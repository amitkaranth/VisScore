# VisScore packages

Optional components live in subfolders so each feature keeps its own dependencies and README.

| Package | Purpose |
|--------|---------|
| [visscore_synthetic](visscore_synthetic/README.md) | Matplotlib + seaborn synthetic Tufte vs chartjunk PNGs (`visscore-generate` CLI). |
| [reddit_scraper](reddit_scraper/README.md) | Scrape chart images from old.reddit.com (no Reddit API / no PRAW). |

The CNN training, Streamlit apps, Plotly/csv tooling, and `src/viscore` inference stack remain at the **repository root**; see the [root README](../README.md).
