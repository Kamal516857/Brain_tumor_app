import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD MODEL & CONFIG ====================
@st.cache_resource
def load_model_and_config():
    """Load pre-trained model and configuration"""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # ---------------- CONFIG ----------------
        config_path = os.path.join(BASE_DIR, 'config.json')

        if not os.path.exists(config_path):
            return None, None, False, f"Config file not found at {config_path}"

        with open(config_path, 'r') as f:
            config = json.load(f)

        # ---------------- MODEL (UPDATED HERE) ----------------
        model_path = os.path.join(BASE_DIR, 'ensemble_model_fixed.h5')

        if not os.path.exists(model_path):
            return None, config, False, f"Model file not found at {model_path}"

        model = load_model(model_path, compile=False)

        return model, config, True, "Model loaded successfully"

    except Exception as e:
        return None, None, False, f"Error loading model: {str(e)}"

# ==================== IMAGE PREPROCESSING ====================
def preprocess_image(image, img_size=224):
    img = image.convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==================== PREDICTION ====================
def predict_tumor(image, model, img_size=224):
    img_array = preprocess_image(image, img_size)
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    return predictions, predicted_idx

# ==================== INTERPRETATION ====================
def get_interpretation(class_name, confidence):
    interpretations = {
        'glioma': 'Glioma tumors arise from glial cells in the brain.',
        'meningioma': 'Meningioma tumors develop in the meninges.',
        'pituitary': 'Pituitary tumors originate in the pituitary gland.',
        'notumor': 'No tumor detected. Scan appears normal.'
    }
    return interpretations.get(class_name.lower(), 'Analysis complete.')

# ==================== MAIN ====================
def main():
    model, config, model_loaded, status_msg = load_model_and_config()

    img_size = config.get('img_size', 224) if config else 224
    class_names = config.get(
        'class_names',
        ['glioma', 'meningioma', 'notumor', 'pituitary']
    ) if config else []

    st.title("🧠 Brain Tumor Detection System")

    if not model_loaded:
        st.error(status_msg)
        return

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze"):
            predictions, idx = predict_tumor(image, model, img_size)

            predicted_class = class_names[idx]
            confidence = predictions[idx] * 100

            st.success(f"Prediction: {predicted_class.upper()}")
            st.write(f"Confidence: {confidence:.2f}%")

            st.info(get_interpretation(predicted_class, confidence))

            # Chart
            fig = go.Figure([go.Bar(
                x=class_names,
                y=predictions * 100
            )])
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
