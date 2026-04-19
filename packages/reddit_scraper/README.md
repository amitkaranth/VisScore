# VisScore Reddit Scraper

Python based web scraper for collecting chart images from **old.reddit.com** (e.g. r/dataisbeautiful, r/dataisugly) to build a machine learning dataset.

**No Reddit API. No PRAW.** Uses only `requests` and BeautifulSoup on static HTML. No API keys or OAuth.

---

## Requirements

- Python 3.9+
- Dependencies in `requirements.txt`

## Install

```bash
cd packages/reddit_scraper
pip install -r requirements.txt
```

## Usage

### CLI (recommended)

```bash
# From packages/reddit_scraper
python main.py [subreddit] [--sort top|hot|new] [--time day|week|month|year|all] [--max-posts N]
```

**Examples:**

```bash
# r/dataisbeautiful, top of month, up to 50 image posts (default)
python main.py

# r/dataisugly, hot, 100 posts
python main.py dataisugly --sort hot --max-posts 100

# r/dataisbeautiful, top of all time, 200 posts
python main.py dataisbeautiful --sort top --time all --max-posts 200

# Disable resolving image from permalink (faster, fewer images)
python main.py dataisbeautiful --no-resolve --max-posts 30

# Disable resume (re-download everything)
python main.py dataisbeautiful --no-resume
```

### Options

| Option | Description |
|--------|-------------|
| `subreddit` | Subreddit name without `r/` (default: `dataisbeautiful`) |
| `--sort` | `top`, `hot`, or `new` (default: `top`) |
| `--time` | For `top` only: `day`, `week`, `month`, `year`, `all` (default: `month`) |
| `--max-posts` | Max image posts to download (default: 50) |
| `--no-resolve` | Do not fetch post page to get image URL when listing only has comments link |
| `--no-resume` | Ignore existing metadata/files and re-run from scratch |
| `--config` | Path to `config.yaml` (optional) |

---

## Output Layout

- **Images:** `data/raw/<subreddit>/` (e.g. `data/raw/dataisbeautiful/`)
- **Metadata:** `data/metadata/<subreddit>_<sort>_<time>.json`
- **Logs:** `logs/scraper.log`

Metadata JSON includes for each image: `post_id`, `title`, `upvotes`, `image_url`, `permalink`, `subreddit`, `local_path`, `width`, `height`, `sha256`.

---

## Behavior

- **Listing:** Fetches old.reddit.com listing pages (HTML), parses with BeautifulSoup, paginates with `after`.
- **Image URLs:** Only direct image links are kept: `.jpg`, `.png`, `.webp`, i.redd.it, single-imgur. Videos, gifs, albums, galleries, and articles are excluded.
- **Resolve:** If the listing only shows a comments link, the scraper can fetch the post page once to read `og:image` (or first i.redd.it link). Use `--no-resolve` to skip this and only keep posts whose listing link is already an image.
- **Download:** Each URL is downloaded with retries and backoff. Response is checked: status code and `Content-Type: image/*`.
- **Validation:** After download, images are opened with PIL; rejected if width or height &lt; 300 px or if corrupted. SHA256 is computed for deduplication.
- **Resume:** By default, posts already present in metadata or as files in `data/raw/<subreddit>/` are skipped; use `--no-resume` to override.
- **Rate limiting:** Configurable delay (default 2–5 s) between requests. No Reddit API, so no API key; be respectful to avoid blocks.

---

## Config

Edit `config.yaml` to change:

- `scraper`: `base_url`, `user_agent`, `request_delay_min/max_seconds`, `max_retries`, `timeout_seconds`
- `image`: `min_width_px`, `min_height_px`
- `paths`: `raw_images_dir`, `metadata_dir`, `logs_dir`
- `resume`: set to `false` to disable resume by default

---

## Code Layout

```
packages/reddit_scraper/
├── scraper/
│   ├── reddit_scraper.py   # Listing fetch + parse (old.reddit.com, BeautifulSoup)
│   ├── image_downloader.py # Download + HTTP/Content-Type + PIL + SHA256
│   ├── validators.py       # Image validation (PIL, dimensions, hash)
│   └── utils.py            # Config, logging, retry, User-Agent, sleep
├── data/
│   ├── raw/
│   │   ├── dataisbeautiful/
│   │   └── dataisugly/
│   └── metadata/
├── logs/
├── config.yaml
├── main.py                 # CLI + orchestration
├── requirements.txt
└── README.md
```

---

## Academic / Evaluation Notes

- No use of PRAW or Reddit’s official API; all data comes from public old.reddit.com HTML.
- User-Agent is set to a descriptive value; Reddit may block generic or missing User-Agents.
- Retries use exponential backoff; failures and skips are logged.
- Design is modular and type-hinted for clarity and unit testing.

---

## Troubleshooting

- **403 Blocked:** Reddit may block requests from some networks (e.g. datacenters, CI). Run from a normal home/office connection. Ensure `User-Agent` in `config.yaml` is set to a descriptive value (e.g. `visscore-research-scraper/1.0 (academic project)`).
- **No images found:** Use `--no-resolve` to see if listing-only image links work; otherwise ensure the subreddit has link posts to i.redd.it or direct image URLs.
