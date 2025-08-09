# app.py
import os
import requests
import streamlit as st

def download_file(url, filename):
    if not os.path.exists(filename):
        st.write(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)

download_file(st.secrets["MODEL_URL_EFF"], "efficientnet_model.h5")
download_file(st.secrets["MODEL_URL_RES"], "resnet_model.h5")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import base64, tempfile, os
import matplotlib.pyplot as plt

# TTS helpers (pyttsx3 optional, fallback gTTS)
use_pyttsx3 = False
try:
    import pyttsx3
    use_pyttsx3 = True
except Exception:
    use_pyttsx3 = False
from gtts import gTTS

# ----------------- Config -----------------
MODELS_DIR = "models"
EFF_MODEL_PATH = os.path.join(MODELS_DIR, "efficientnet_model.h5")
RES_MODEL_PATH = os.path.join(MODELS_DIR, "resnet_model.h5")
CONF_MAT_PATH = os.path.join(MODELS_DIR, "confusion_matrix_efficientnet.png")
SAMPLE_PRED_PATH = os.path.join(MODELS_DIR, "sample_predictions.png")
TRAIN_ACC_PATH = os.path.join(MODELS_DIR, "training_accuracy.png")
TRAIN_LOSS_PATH = os.path.join(MODELS_DIR, "training_loss.png")

IMG_SIZE = (224, 224)
CLASS_NAMES = None  # Will attempt to infer from models or user can edit below

# If you know exact classes replace here, e.g.:
# CLASS_NAMES = ['daisy','dandelion','rose','sunflower','tulip']

# ----------------- Load models -----------------
@st.cache_resource(show_spinner=False)
def load_models():
    eff, res = None, None
    if os.path.exists(EFF_MODEL_PATH):
        eff = tf.keras.models.load_model(EFF_MODEL_PATH)
    if os.path.exists(RES_MODEL_PATH):
        res = tf.keras.models.load_model(RES_MODEL_PATH)
    return eff, res

eff_model, res_model = load_models()

# Attempt to infer class names from data in models (if available) else fallback
if CLASS_NAMES is None:
    # try to get from models' output layer names if saved as metadata
    try:
        if eff_model is not None:
            out_shape = eff_model.output_shape
            # No direct labels possible — keep generic
    except Exception:
        pass
    # final fallback: short generic labels (user should update if needed)
    CLASS_NAMES = ["Class 0", "Class 1"] if (eff_model is None or (eff_model.output_shape[-1] == 2)) else \
                  [f"Class {i}" for i in range(eff_model.output_shape[-1])] if eff_model is not None else ["Class 0","Class 1"]

# ----------------- Page style -----------------
st.set_page_config(page_title="Next-Gen AI Classifier", page_icon="🤖", layout="centered")
st.markdown("""
<style>
    body { background-color: #0b0f12; color: #E6F2F1; font-family: 'Inter', sans-serif; }
    .stButton>button { background: linear-gradient(90deg,#00f7ff,#00ffaa); color:black; font-weight:700; }
    .card { background: rgba(255,255,255,0.03); border-radius:12px; padding:14px; }
    .muted { color:#9aaeb0; font-size:0.95rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:#00f7ff'>SK Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center' class='muted'>Built by Sameen khan</p>", unsafe_allow_html=True)
st.divider()

# ----------------- Utility functions -----------------
def preprocess_pil(img: Image.Image):
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr]*3, axis=-1)
    return np.expand_dims(arr, axis=0)

def ensemble_predict(img_pil):
    """Return averaged probabilities from available models"""
    x = preprocess_pil(img_pil)
    preds = []
    if eff_model is not None:
        preds.append(eff_model.predict(x)[0])
    if res_model is not None:
        preds.append(res_model.predict(x)[0])
    if len(preds) == 0:
        raise RuntimeError("No models found in models/ directory.")
    avg = np.mean(preds, axis=0)
    return avg

def speak_prediction(text: str):
    # Try pyttsx3 (offline male voice), else gTTS fallback
    if use_pyttsx3:
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            # Try to pick a male voice if available
            for v in voices:
                if 'male' in (v.name or '').lower() or 'm' in (v.id or '').lower():
                    engine.setProperty('voice', v.id)
                    break
            engine.setProperty('rate', 150)
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            tmp_path = tmpf.name; tmpf.close()
            engine.save_to_file(text, tmp_path); engine.runAndWait()
            with open(tmp_path, 'rb') as f:
                b = f.read()
            os.remove(tmp_path)
            b64 = base64.b64encode(b).decode()
            st.markdown(f'<audio autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>', unsafe_allow_html=True)
            return
        except Exception as e:
            print("pyttsx3 error:", e)
    # gTTS fallback
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tmp_path = tmpf.name; tmpf.close()
        tts.save(tmp_path)
        with open(tmp_path, 'rb') as f:
            b = f.read()
        os.remove(tmp_path)
        b64 = base64.b64encode(b).decode()
        st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)
    except Exception as e:
        print("gTTS error:", e)

# ----------------- Sidebar / controls -----------------
with st.sidebar:
    st.header("Controls")
    show_conf = st.checkbox("Show Evaluation Images (confusion / samples)", value=True)
    show_graphs = st.checkbox("Show Training Graphs", value=True)
    speak_toggle = st.checkbox("Enable Voice Output", value=True)
    st.markdown("---")
    # Model selection
    chosen_models = st.multiselect("Use models (ensemble)", 
                                   options=["EfficientNet", "ResNet"],
                                   default=["EfficientNet","ResNet"] if (eff_model is not None and res_model is not None) else (["EfficientNet"] if eff_model is not None else ["ResNet"]))
    st.markdown("---")
    st.markdown("Tip: For best demo, use test images from `dataset_split/test`")

# Dynamically set which models are used by toggling global references
def get_active_models():
    active = []
    if "EfficientNet" in chosen_models and eff_model is not None:
        active.append(eff_model)
    if "ResNet" in chosen_models and res_model is not None:
        active.append(res_model)
    return active

# ----------------- Mode selection -----------------
mode = st.radio("Mode:", ["Upload Image", "Live Webcam"], horizontal=True)

if mode == "Upload Image":
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        # Predict
        try:
            avg = ensemble_predict(img)
            # top-3
            top_idx = np.argsort(avg)[::-1][:3]
            st.markdown("### Prediction")
            st.markdown(f"**Top:** {CLASS_NAMES[top_idx[0]]} — {avg[top_idx[0]]*100:.2f}%")
            with st.expander("Top 3 probabilities"):
                for i in top_idx:
                    st.write(f"- {CLASS_NAMES[i]} : {avg[i]*100:.2f}%")
            # chart
            fig = px.bar(x=CLASS_NAMES, y=avg, labels={'x':'Class','y':'Probability'}, text=[f"{p*100:.2f}%" for p in avg],
                         color=avg, color_continuous_scale=["#00f7ff","#00ffaa"])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)
            # speak
            if speak_toggle:
                speak_prediction(f"Predicted class is {CLASS_NAMES[top_idx[0]]} with {avg[top_idx[0]]*100:.1f} percent confidence.")
        except Exception as e:
            st.error("Prediction failed: " + str(e))

elif mode == "Live Webcam":
    st.markdown("### Capture from webcam")
    camera_image = st.camera_input("Capture")
    if camera_image is not None:
        img = Image.open(camera_image).convert("RGB")
        st.image(img, caption="Captured Image", use_column_width=True)
        try:
            avg = ensemble_predict(img)
            top_idx = np.argsort(avg)[::-1][:3]
            st.markdown("### Prediction")
            st.markdown(f"**Top:** {CLASS_NAMES[top_idx[0]]} — {avg[top_idx[0]]*100:.2f}%")
            with st.expander("Top 3 probabilities"):
                for i in top_idx:
                    st.write(f"- {CLASS_NAMES[i]} : {avg[i]*100:.2f}%")
            fig = px.bar(x=CLASS_NAMES, y=avg, labels={'x':'Class','y':'Probability'}, text=[f"{p*100:.2f}%" for p in avg],
                         color=avg, color_continuous_scale=["#00f7ff","#00ffaa"])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)
            if speak_toggle:
                speak_prediction(f"Predicted class is {CLASS_NAMES[top_idx[0]]} with {avg[top_idx[0]]*100:.1f} percent confidence.")
        except Exception as e:
            st.error("Prediction failed: " + str(e))

st.divider()

# ----------------- Show evaluation images if requested -----------------
if show_conf:
    st.markdown("## Evaluation & Samples")
    cols = st.columns(2)
    if os.path.exists(CONF_MAT_PATH):
        cols[0].image(CONF_MAT_PATH, caption="Confusion Matrix (EfficientNet)", use_column_width=True)
    else:
        cols[0].info("Confusion matrix image not found (run train.py to generate).")
    if os.path.exists(SAMPLE_PRED_PATH):
        cols[1].image(SAMPLE_PRED_PATH, caption="Sample Predictions", use_column_width=True)
    else:
        cols[1].info("Sample predictions image not found (run train.py to generate).")

if show_graphs:
    st.markdown("## Training Graphs")
    gcols = st.columns(2)
    if os.path.exists(TRAIN_ACC_PATH):
        gcols[0].image(TRAIN_ACC_PATH, caption="Training Accuracy", use_column_width=True)
    else:
        gcols[0].info("Training accuracy image not found.")
    if os.path.exists(TRAIN_LOSS_PATH):
        gcols[1].image(TRAIN_LOSS_PATH, caption="Training Loss", use_column_width=True)
    else:
        gcols[1].info("Training loss image not found.")

st.markdown("---")
st.markdown("⚡ Powered by TensorFlow & Streamlit | Designed by Sameen")
