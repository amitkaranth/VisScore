"""
VisScore: Streamlit Web UI for Visualization Quality Classification
Allows users to upload chart images and get quality predictions using trained models.

Usage:
  streamlit run app.py
"""

import os
import json
import tempfile
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import inference functions
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import load_model, predict, save_gradcam


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VisScore",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def get_device():
    """Get available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_available_models(results_dir="./results"):
    """Find all trained models in results directory."""
    models_info = {}
    if os.path.exists(results_dir):
        for config_file in os.listdir(results_dir):
            if config_file.endswith("training_config.json"):
                config_path = os.path.join(results_dir, config_file)
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    model_name = config.get("model", "unknown")
                    model_path = os.path.join(results_dir, "best_model.pth")
                    if os.path.exists(model_path):
                        models_info[f"{model_name}"] = {
                            "path": model_path,
                            "model_name": model_name,
                            "epochs": config.get("epochs", "N/A"),
                            "batch_size": config.get("batch_size", "N/A"),
                            "val_accuracy": config.get("best_val_accuracy", "N/A"),
                        }
                except Exception as e:
                    st.warning(f"Could not load config {config_file}: {e}")
    
    # If no models found, check for default model
    if not models_info:
        default_model_path = "./results/best_model.pth"
        if os.path.exists(default_model_path):
            models_info["ResNet50 (Default)"] = {
                "path": default_model_path,
                "model_name": "resnet50",
                "epochs": "N/A",
                "batch_size": "N/A",
                "val_accuracy": "N/A",
            }
    
    return models_info


@st.cache_resource
def load_model_cached(model_path, model_name, device):
    """Load and cache model."""
    return load_model(model_path, model_name, device)


def display_prediction_card(result, input_tensor, original_img, model_name, device):
    """Display prediction results in a nice card format."""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Result")
        label = result["label"]
        confidence = result["confidence"]
        probability = result["probability"]
        
        # Color coding
        if label == "GOOD":
            st.success(f"### ✅ {label}")
            color = "green"
        else:
            st.error(f"### ❌ {label}")
            color = "red"
        
        # Metrics
        st.metric("Confidence", f"{confidence:.1%}")
        st.metric("Probability", f"{probability:.4f}")
    
    with col2:
        st.subheader("Visualization")
        # Display original uploaded image
        st.image(original_img, use_column_width=True, caption="Uploaded Chart")
    
    # Grad-CAM visualization
    st.subheader("Model Attention (Grad-CAM)")
    try:
        from inference import GradCAM
        import cv2
        
        # Get target layer
        if "resnet" in model_name.lower():
            model = load_model_cached(models_info[selected_model]["path"], model_name, device)
            target_layer = model.layer4[-1]
        elif "efficientnet" in model_name.lower():
            model = load_model_cached(models_info[selected_model]["path"], model_name, device)
            target_layer = model.features[-1]
        elif "vgg" in model_name.lower():
            model = load_model_cached(models_info[selected_model]["path"], model_name, device)
            target_layer = model.features[-1]
        else:
            return
        
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(input_tensor)
        
        # Prepare images for display
        img_np = np.array(original_img.resize((224, 224))) / 255.0
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Original
        ax1.imshow(img_np)
        ax1.set_title("Original Chart", fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # Heatmap
        ax2.imshow(cam_resized, cmap='jet')
        ax2.set_title("Grad-CAM Heatmap", fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # Overlay
        ax3.imshow(img_np)
        ax3.imshow(cam_resized, cmap='jet', alpha=0.5)
        ax3.set_title(f"Prediction: {label} ({confidence:.1%})", fontsize=11, fontweight='bold', color=color)
        ax3.axis('off')
        
        plt.suptitle("VisScore: Model Attention Analysis", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        st.pyplot(fig)
    except Exception as e:
        st.info(f"Grad-CAM visualization not available: {e}")


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Title
    st.title("📊 VisScore: Visualization Quality Classifier")
    st.markdown("Predict whether your chart follows Tufte's principles of data visualization")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Find available models
        models_info = find_available_models()
        
        if not models_info:
            st.error("❌ No trained models found in `./results/` directory. Please train a model first.")
            st.stop()
        
        # Model selection
        st.subheader("Model Selection")
        global selected_model
        selected_model = st.selectbox(
            "Select a Model",
            options=list(models_info.keys()),
            help="Choose which trained model to use for prediction"
        )
        
        model_info = models_info[selected_model]
        with st.expander("Model Details"):
            st.json({
                "Model Type": model_info["model_name"],
                "Path": model_info["path"],
                "Validation Accuracy": f"{model_info['val_accuracy']}",
                "Epochs Trained": f"{model_info['epochs']}",
                "Batch Size": f"{model_info['batch_size']}",
            })
        
        st.divider()
        st.subheader("Device Info")
        device = get_device()
        st.info(f"Using: **{device}**")
    
    # Main content
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        st.subheader("📈 Upload Chart Image")
        uploaded_file = st.file_uploader(
            "Choose a chart image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            help="Upload a visualization/chart image to analyze"
        )
    
    with col_info:
        st.subheader("ℹ️ About VisScore")
        st.markdown("""
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
        """)
    
    # Prediction
    if uploaded_file is not None:
        st.divider()
        
        # Load model
        try:
            device = get_device()
            model_name = model_info["model_name"]
            model = load_model_cached(model_info["path"], model_name, device)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
        
        # Save uploaded file temporarily and predict
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            # Get prediction
            with st.spinner("🔍 Analyzing chart..."):
                result, input_tensor, original_img = predict(model, tmp_path, device)
            
            # Display results
            display_prediction_card(result, input_tensor, original_img, model_name, device)
            
            # Additional info
            st.divider()
            st.subheader("📋 Detailed Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Label", result["label"])
            
            with col2:
                st.metric("Probability", f"{result['probability']:.4f}")
            
            with col3:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            
            with col4:
                st.metric("Image", result["image"])
            
            # Clean up
            os.unlink(tmp_path)
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            import traceback
            st.error(traceback.format_exc())
    
    else:
        st.info("👆 Upload a chart image to get started!")
        
        # Example section
        st.divider()
        st.subheader("💡 Example Charts")
        st.markdown("""
        Try uploading:
        - **Good Charts**: bar charts, line plots, scatter plots with clear labels
        - **Bad Charts**: pie charts with many slices, 3D charts, charts with excessive decoration
        """)


if __name__ == "__main__":
    main()
