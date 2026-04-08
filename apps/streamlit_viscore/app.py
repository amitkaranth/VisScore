from __future__ import annotations

import os
import json
import tempfile
import sys
from pathlib import Path

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _bootstrap_src

_bootstrap_src.bootstrap()

from viscore.inference import GradCAM, load_model, predict
from viscore.vlm import combine_cnn_and_vlm, judge_chart_vlm


st.set_page_config(
    page_title="VisScore",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main { background-color: #f8f9fa; }
    .stAlert { padding: 1rem; border-radius: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_available_models(results_dir: str = "./results"):
    models_info: dict[str, dict] = {}
    if not os.path.isdir(results_dir):
        return models_info

    pairs: list[tuple[str, str, str]] = []

    main_cfg = os.path.join(results_dir, "training_config.json")
    main_w = os.path.join(results_dir, "best_model.pth")
    if os.path.isfile(main_cfg) and os.path.isfile(main_w):
        pairs.append((main_cfg, main_w, "active"))

    for fn in sorted(os.listdir(results_dir)):
        if not fn.startswith("training_config_") or not fn.endswith(".json"):
            continue
        tag = fn[len("training_config_") : -len(".json")]
        if not tag:
            continue
        cfg_path = os.path.join(results_dir, fn)
        w_path = os.path.join(results_dir, f"best_model_{tag}.pth")
        if os.path.isfile(w_path):
            pairs.append((cfg_path, w_path, tag))

    seen: set[tuple[str, str]] = set()
    for cfg_path, w_path, suffix in pairs:
        key = (cfg_path, w_path)
        if key in seen:
            continue
        seen.add(key)
        try:
            with open(cfg_path, "r") as f:
                config = json.load(f)
            model_name = config.get("model", "unknown")
            label = f"{model_name} ({suffix})"
            models_info[label] = {
                "path": w_path,
                "model_name": model_name,
                "epochs": config.get("epochs", "N/A"),
                "batch_size": config.get("batch_size", "N/A"),
                "val_accuracy": config.get("best_val_accuracy", "N/A"),
            }
        except Exception as e:
            st.warning(f"Could not load config {cfg_path}: {e}")

    if not models_info:
        fallback = os.path.join(results_dir, "best_model.pth")
        if os.path.isfile(fallback):
            models_info["ResNet50 (Default)"] = {
                "path": fallback,
                "model_name": "resnet50",
                "epochs": "N/A",
                "batch_size": "N/A",
                "val_accuracy": "N/A",
            }

    return models_info


@st.cache_resource
def load_model_cached(model_path: str, model_name: str, device):
    return load_model(model_path, model_name, device)


def display_prediction_card(result, input_tensor, original_img, model_name, device, model_path):
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Result")
        label = result["label"]
        confidence = result["confidence"]
        probability = result["probability"]

        if label == "GOOD":
            st.success(f"### ✅ {label}")
            color = "green"
        else:
            st.error(f"### ❌ {label}")
            color = "red"

        st.metric("Confidence", f"{confidence:.1%}")
        st.metric("Probability", f"{probability:.4f}")

    with col2:
        st.subheader("Visualization")
        st.image(original_img, use_container_width=True, caption="Uploaded Chart")

    st.subheader("Model Attention (Grad-CAM)")
    try:
        import cv2

        model = load_model_cached(model_path, model_name, device)
        if "resnet" in model_name.lower():
            target_layer = model.layer4[-1]
        elif "efficientnet" in model_name.lower():
            target_layer = model.features[-1]
        elif "vgg" in model_name.lower():
            target_layer = model.features[-1]
        else:
            st.info("Grad-CAM: unsupported architecture.")
            return

        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(input_tensor)

        img_np = np.array(original_img.resize((224, 224))) / 255.0
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        ax1.imshow(img_np)
        ax1.set_title("Original Chart", fontsize=11, fontweight="bold")
        ax1.axis("off")

        ax2.imshow(cam_resized, cmap="jet")
        ax2.set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold")
        ax2.axis("off")

        ax3.imshow(img_np)
        ax3.imshow(cam_resized, cmap="jet", alpha=0.5)
        ax3.set_title(f"Prediction: {label} ({confidence:.1%})", fontsize=11, fontweight="bold", color=color)
        ax3.axis("off")

        plt.suptitle("VisScore: Model Attention Analysis", fontsize=12, fontweight="bold")
        plt.tight_layout()

        st.pyplot(fig)
    except Exception as e:
        st.info(f"Grad-CAM visualization not available: {e}")


def main():
    st.title("📊 VisScore: Visualization Quality Classifier")
    st.markdown("Predict whether your chart follows Tufte's principles of data visualization")

    with st.sidebar:
        st.header("⚙️ Configuration")

        models_info = find_available_models()
        if not models_info:
            st.error("❌ No trained models found in `./results/` directory. Please train a model first.")
            st.stop()

        st.subheader("Model Selection")
        selected_model = st.selectbox("Select a Model", options=list(models_info.keys()))
        model_info = models_info[selected_model]

        with st.expander("Model Details"):
            st.json(
                {
                    "Model Type": model_info["model_name"],
                    "Path": model_info["path"],
                    "Validation Accuracy": f"{model_info['val_accuracy']}",
                    "Epochs Trained": f"{model_info['epochs']}",
                    "Batch Size": f"{model_info['batch_size']}",
                }
            )

        st.divider()
        st.subheader("Multimodal (VLM + consensus)")
        enable_vlm = st.checkbox("Enable VLM + consensus", value=False)
        vlm_provider_ui = st.selectbox(
            "VLM provider",
            options=["gemini", "groq"],
            index=0 if (os.environ.get("VLM_PROVIDER", "gemini").lower() != "groq") else 1,
        )
        gemini_key = st.text_input("Gemini API key", type="password")
        groq_key = st.text_input("Groq API key", type="password")
        _gemini_m = os.environ.get("GEMINI_VLM_MODEL") or "gemini-flash-latest"
        _groq_m = os.environ.get("GROQ_VLM_MODEL") or "meta-llama/llama-4-scout-17b-16e-instruct"
        vlm_model = st.text_input("VLM model id (optional)", value=_groq_m if vlm_provider_ui == "groq" else _gemini_m)

        st.divider()
        device = get_device()
        st.info(f"Using: **{device}**")

    col_upload, col_info = st.columns([2, 1])
    with col_upload:
        st.subheader("📈 Upload Chart Image")
        uploaded_file = st.file_uploader("Choose a chart image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    with col_info:
        st.subheader("ℹ️ About VisScore")
        st.markdown(
            """
        **Good Charts** follow Tufte's principles:
        - High data-ink ratio
        - Minimal chartjunk
        - Clear labels
        - Appropriate visualization type

        **Bad Charts** violate one or more principles:
        - Low data-ink ratio
        - Excessive decoration
        - Misleading axes
        - Inappropriate visualization
        """
        )

    if uploaded_file is None:
        st.info("👆 Upload a chart image to get started!")
        return

    st.divider()
    device = get_device()
    model_name = model_info["model_name"]
    model_path = model_info["path"]
    try:
        model = load_model_cached(model_path, model_name, device)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        with st.spinner("🔍 Analyzing chart (CNN)..."):
            result, input_tensor, original_img = predict(model, tmp_path, device)

        display_prediction_card(result, input_tensor, original_img, model_name, device, model_path)

        g_key = (gemini_key or "").strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        q_key = (groq_key or "").strip() or os.environ.get("GROQ_API_KEY")
        if enable_vlm:
            st.divider()
            st.subheader("🧠 VLM (Tufte reasoning)")
            need_warn = (vlm_provider_ui == "gemini" and not g_key) or (vlm_provider_ui == "groq" and not q_key)
            if need_warn:
                st.warning(f"Add a **{vlm_provider_ui}** API key in the sidebar (or set env).")
            else:
                with st.spinner(f"Calling {vlm_provider_ui} (vision)..."):
                    vlm_out = judge_chart_vlm(
                        tmp_path,
                        provider=vlm_provider_ui,
                        gemini_api_key=g_key or None,
                        groq_api_key=q_key or None,
                        model_name=(vlm_model or "").strip() or None,
                    )
                if vlm_out.get("error"):
                    st.error(vlm_out["error"])
                else:
                    v = vlm_out.get("verdict", "?")
                    (st.success if v == "GOOD" else st.error)(f"**VLM verdict:** {v}")
                    st.markdown(f"**Reasoning:** {vlm_out.get('reasoning', '')}")
                    st.caption(f"Model: {vlm_out.get('model', '')}")

                cons = combine_cnn_and_vlm(result["label"], vlm_out)
                st.divider()
                st.subheader("⚖️ Consensus")
                fv = cons.get("final_verdict")
                if fv == "SPLIT":
                    st.warning(f"**Split decision** — CNN: **{cons.get('cnn_vote')}**, VLM: **{cons.get('vlm_vote')}**.")
                elif fv == "GOOD":
                    st.success(f"**Final:** {fv} ({cons.get('agreement')})")
                else:
                    st.error(f"**Final:** {fv} ({cons.get('agreement')})")
                if cons.get("agreement") == "cnn_only":
                    st.info(cons.get("note", ""))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()

