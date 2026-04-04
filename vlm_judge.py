"""
VisScore: Vision-language judge (Tufte-style) for multimodal consensus with the CNN.

Providers (set VLM_PROVIDER=gemini or groq):

- **Gemini** (default): https://aistudio.google.com/apikey — GEMINI_API_KEY
  Model IDs change often; we try your choice first, then a fallback list
  (e.g. gemini-flash-latest) if you get 404/429.

- **Groq** (free tier): https://console.groq.com/keys — GROQ_API_KEY
  Vision models use the OpenAI-compatible chat API with a base64 image.

Returns structured verdict + reasoning.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any

import requests
from PIL import Image

# Shorter names first = try stable “latest” aliases before deprecated ids.
_GEMINI_MODEL_FALLBACKS = [
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-8b",
    "gemini-2.5-flash-preview-05-20",
]

TUFTE_VLM_PROMPT = """You are a strict judge of static charts using Edward Tufte's principles. Your job is to match how a "good vs bad visualization" classifier would label classroom / textbook examples—not to be generous.

**Label BAD if ANY of these are clearly visible** (even if the numbers are still readable):
- Very heavy, dense, or decorative grid lines (especially many more grid lines than needed, or grids that dominate the plot).
- Strong non-neutral plot background (solid color fills, tinted panels, stripes, textures) where most ink is not the data.
- Obvious low data-ink ratio: decoration, borders, shading, or grid compete with the bars/lines/points for attention.
- Chartjunk: 3D effects, pictorial gimmicks, excessive labels, rainbow coloring for no reason, dual axes used to imply false relationships.
- Misleading or missing baseline / truncated axis that exaggerates differences.
- Any title or caption in the image that describes the graphic as "poor", "bad", "misleading", or mentions "low data-ink" / "poor data-ink ratio" — treat that as evidence the design is intentionally flawed; verdict BAD.

**Label GOOD only if ALL are true:**
- The data marks (bars, lines, points, pie slices, etc.) carry most of the visual weight.
- Gridlines are absent or very light and sparse; background is white or near-white.
- No serious Tufte violations above.

**Tie-break:** If you are unsure, choose BAD (strict). GOOD must mean "clean Tufte-style presentation," not merely "I can read the values."

Look at the entire image including titles and captions.

Respond with ONLY a single JSON object (no markdown fences, no other text) using exactly this shape:
{"verdict": "GOOD", "reasoning": "your explanation in 2-5 sentences"}

Rules:
- "verdict" must be exactly GOOD or BAD (uppercase).
- "reasoning" must cite specific visible elements (grid, background, caption text, etc.).
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    # Fallback: first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _normalize_verdict(raw: str) -> str:
    s = (raw or "").strip().upper()
    if s in ("GOOD", "G"):
        return "GOOD"
    if s in ("BAD", "B"):
        return "BAD"
    if "GOOD" in s and "BAD" not in s:
        return "GOOD"
    if "BAD" in s:
        return "BAD"
    return "BAD"


def _gemini_model_candidates(primary: str | None) -> list[str]:
    primary = (primary or os.environ.get("GEMINI_VLM_MODEL") or "gemini-flash-latest").strip()
    out: list[str] = []
    seen: set[str] = set()
    for mid in [primary, *_GEMINI_MODEL_FALLBACKS]:
        if mid and mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


def judge_chart_gemini(
    image_path: str,
    api_key: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Call Gemini with the chart image. Tries multiple model IDs on 404/429-style failures.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return {
            "verdict": None,
            "reasoning": "",
            "model": None,
            "error": "Missing API key. Set GEMINI_API_KEY (get a free key at https://aistudio.google.com/apikey).",
        }

    try:
        import google.generativeai as genai
    except ImportError:
        return {
            "verdict": None,
            "reasoning": "",
            "model": None,
            "error": "Install google-generativeai: pip install google-generativeai",
        }

    genai.configure(api_key=key)
    img = Image.open(image_path).convert("RGB")
    candidates = _gemini_model_candidates(model_name)
    errors: list[str] = []
    used_model: str | None = None

    for mid in candidates:
        try:
            model = genai.GenerativeModel(mid)
            response = model.generate_content(
                [TUFTE_VLM_PROMPT, img],
                generation_config={"temperature": 0.1, "max_output_tokens": 1024},
            )
        except Exception as e:
            err = str(e)
            errors.append(f"{mid}: {err}")
            if "404" in err or "not found" in err.lower() or "429" in err or "quota" in err.lower():
                continue
            return {
                "verdict": None,
                "reasoning": "",
                "model": mid,
                "error": f"Gemini API error: {e}",
            }

        used_model = mid
        text = ""
        try:
            text = response.text or ""
        except Exception:
            for cand in getattr(response, "candidates", []) or []:
                parts = getattr(cand.content, "parts", None) or []
                for p in parts:
                    if getattr(p, "text", None):
                        text += p.text
        text = (text or "").strip()
        if not text:
            errors.append(f"{mid}: empty response")
            continue

        try:
            data = _extract_json_object(text)
            verdict = _normalize_verdict(str(data.get("verdict", "")))
            reasoning = str(data.get("reasoning", "")).strip()
            if not reasoning:
                reasoning = text[:2000]
            return {
                "verdict": verdict,
                "reasoning": reasoning,
                "model": used_model,
                "raw_text": text,
            }
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            return {
                "verdict": None,
                "reasoning": text[:2000],
                "model": used_model,
                "error": f"Could not parse JSON from model: {e}",
                "raw_text": text,
            }

    return {
        "verdict": None,
        "reasoning": "",
        "model": candidates[0] if candidates else None,
        "error": "Gemini: all model attempts failed. Last tries:\n"
        + "\n".join(errors[-5:]),
    }


def _groq_jpeg_data_url(image_path: str, max_base64_chars: int = 3_500_000) -> str:
    """
    Groq base64 image limit ~4MB decoded; keep encoded payload under max_base64_chars.
    Resize + lower JPEG quality until small enough.
    """
    img = Image.open(image_path).convert("RGB")
    max_side = 1536
    quality = 85
    for _ in range(12):
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        raw = buf.getvalue()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        if len(b64) <= max_base64_chars:
            return f"data:image/jpeg;base64,{b64}"
        max_side = max(256, int(max_side * 0.75))
        quality = max(50, quality - 8)
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# Groq vision: use current docs model id; older llama-3.2-vision ids often return 400.
_GROQ_VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
]


def judge_chart_groq(
    image_path: str,
    api_key: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Groq OpenAI-compatible vision API (free tier). Set GROQ_API_KEY.
    Default model: meta-llama/llama-4-scout-17b-16e-instruct (see Groq vision docs).
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        return {
            "verdict": None,
            "reasoning": "",
            "model": None,
            "error": "Missing GROQ_API_KEY. Free key: https://console.groq.com/keys",
        }

    primary = (model_name or os.environ.get("GROQ_VLM_MODEL") or "").strip()
    if primary and "gemini" in primary.lower():
        primary = ""  # UI default may be a Gemini id when provider is Groq
    candidates = []
    if primary:
        candidates.append(primary)
    for m in _GROQ_VISION_MODELS:
        if m not in candidates:
            candidates.append(m)

    try:
        data_url = _groq_jpeg_data_url(image_path)
    except Exception as e:
        return {
            "verdict": None,
            "reasoning": "",
            "model": candidates[0],
            "error": f"Could not prepare image for Groq: {e}",
        }

    last_err = ""
    text = ""
    used_mid = ""

    for mid in candidates:
        payload = {
            "model": mid,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TUFTE_VLM_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_completion_tokens": 1024,
        }

        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
            if not r.ok:
                try:
                    detail = r.json()
                except Exception:
                    detail = (r.text or "")[:500]
                last_err = f"{r.status_code} {detail}"
                if r.status_code in (400, 404):
                    continue
                return {
                    "verdict": None,
                    "reasoning": "",
                    "model": mid,
                    "error": f"Groq API error: {last_err}",
                }
            body = r.json()
            text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            used_mid = mid
            break
        except requests.RequestException as e:
            return {
                "verdict": None,
                "reasoning": "",
                "model": mid,
                "error": f"Groq API error: {e}",
            }

    if not used_mid:
        return {
            "verdict": None,
            "reasoning": "",
            "model": candidates[0] if candidates else None,
            "error": f"Groq: all model attempts failed. Last: {last_err}",
        }

    text = text.strip()
    if not text:
        return {
            "verdict": None,
            "reasoning": "",
            "model": used_mid,
            "error": "Empty response from Groq.",
        }

    try:
        data = _extract_json_object(text)
        verdict = _normalize_verdict(str(data.get("verdict", "")))
        reasoning = str(data.get("reasoning", "")).strip()
        if not reasoning:
            reasoning = text[:2000]
        return {"verdict": verdict, "reasoning": reasoning, "model": used_mid, "raw_text": text}
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return {
            "verdict": None,
            "reasoning": text[:2000],
            "model": used_mid,
            "error": f"Could not parse JSON from model: {e}",
            "raw_text": text,
        }


def judge_chart_vlm(
    image_path: str,
    provider: str | None = None,
    gemini_api_key: str | None = None,
    groq_api_key: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Dispatch to Gemini or Groq based on VLM_PROVIDER env or `provider` argument
    ("gemini" | "groq").
    """
    p = (provider or os.environ.get("VLM_PROVIDER", "gemini")).strip().lower()
    if p == "groq":
        return judge_chart_groq(image_path, api_key=groq_api_key, model_name=model_name)
    return judge_chart_gemini(image_path, api_key=gemini_api_key, model_name=model_name)


def combine_cnn_and_vlm(
    cnn_label: str,
    vlm_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Majority over {CNN, VLM}. With two voters: both agree -> that label; split -> SPLIT.
    """
    cnn = "GOOD" if str(cnn_label).upper() == "GOOD" else "BAD"
    vlm_verdict = vlm_result.get("verdict")
    if vlm_verdict is None:
        return {
            "final_verdict": cnn,
            "agreement": "cnn_only",
            "cnn_vote": cnn,
            "vlm_vote": None,
            "note": vlm_result.get("error", "VLM unavailable"),
        }

    vlm = str(vlm_verdict).upper()
    if vlm not in ("GOOD", "BAD"):
        vlm = "BAD"

    if cnn == vlm:
        return {
            "final_verdict": cnn,
            "agreement": "unanimous",
            "cnn_vote": cnn,
            "vlm_vote": vlm,
        }

    return {
        "final_verdict": "SPLIT",
        "agreement": "disagree",
        "cnn_vote": cnn,
        "vlm_vote": vlm,
        "note": "No majority (1–1). Review CNN score and VLM reasoning below.",
    }
