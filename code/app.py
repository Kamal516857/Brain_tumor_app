import tensorflow as tf
import streamlit as st
st.write("TF VERSION:", tf.__version__)
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from keras.models import load_model
import numpy as np
from PIL import Image
import json
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

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
        /* Root variables */
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #0a0;
            --danger-color: #c00;
            --light-bg: #f8f9ff;
            --border-color: #e0e0ff;
        }
        
        /* Main container */
        .main {
            padding: 1rem;
        }
        
        /* Header styling */
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 1rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .header-container h1 {
            font-size: clamp(1.5rem, 5vw, 2.5rem);
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        
        .header-container p {
            font-size: clamp(0.9rem, 3vw, 1.1rem);
            opacity: 0.9;
            margin: 0;
        }
        
        /* Card styling */
        .metric-card {
            background-color: #f8f9ff;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* File uploader styling */
        .stFileUploader {
            border: 2px dashed #667eea !important;
            border-radius: 10px !important;
            padding: 2rem !important;
        }
        
        /* Alert styling */
        .success-alert {
            background-color: #e0ffe0;
            color: #0a0;
            padding: 1.25rem;
            border-radius: 8px;
            border-left: 5px solid #0a0;
            font-weight: 500;
        }
        
        .danger-alert {
            background-color: #ffe0e0;
            color: #c00;
            padding: 1.25rem;
            border-radius: 8px;
            border-left: 5px solid #c00;
            font-weight: 500;
        }
        
        .info-alert {
            background-color: #e0f0ff;
            color: #0066cc;
            padding: 1.25rem;
            border-radius: 8px;
            border-left: 5px solid #0066cc;
            font-weight: 500;
        }
        
        .warning-alert {
            background-color: #fff5e0;
            color: #cc6600;
            padding: 1.25rem;
            border-radius: 8px;
            border-left: 5px solid #cc6600;
            font-weight: 500;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main {
                padding: 0.5rem;
            }
            
            .header-container {
                padding: 1.5rem 1rem;
                margin-bottom: 1.5rem;
            }
            
            .metric-card {
                padding: 1rem;
                margin-bottom: 1rem;
            }
            
            .stButton > button {
                font-size: 0.9rem;
                padding: 0.6rem;
            }
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            padding: 1rem 0.5rem;
        }
        
        /* Divider */
        hr {
            margin: 1.5rem 0;
            border: none;
            border-top: 2px solid #e0e0ff;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL & CONFIG ====================
@st.cache_resource
def load_model_and_config():
    """Load pre-trained model and configuration"""
    try:
        # Get current file directory
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # ---------------- CONFIG ----------------
        config_path = os.path.join(BASE_DIR, 'config.json')

        if not os.path.exists(config_path):
            return None, None, False, f"Config file not found at {config_path}"

        with open(config_path, 'r') as f:
            config = json.load(f)

        # ---------------- MODEL ----------------
        model_path = os.path.join(BASE_DIR, 'ensemble_model_fixed.h5') 
        
        st.write("Loading model from:", model_path)
        st.write("File exists:", os.path.exists(model_path))
        if os.path.exists(model_path):
            st.write("File size:", os.path.getsize(model_path), "bytes")
        if not os.path.exists(model_path):
            return None, config, False, f"Model file not found at {model_path}"
        model = tf.keras.models.load_model( model_path, compile=False,
                                           custom_objects={ "InputLayer": tf.keras.layers.InputLayer } )
        return model, config, True, "Model loaded successfully"

    except Exception as e:
        return None, None, False, f"Error loading model: {str(e)}"

# ==================== IMAGE PREPROCESSING ====================
def preprocess_image(image, img_size=224):
    """Preprocess image for prediction"""
    try:
        img = image.convert('RGB')
        img = img.resize((img_size, img_size))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")

# ==================== PREDICTION ====================
def predict_tumor(image, model, img_size=224):
    """Make prediction on image"""
    img_array = preprocess_image(image, img_size)
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    return predictions, predicted_idx

# ==================== INTERPRETATION ====================
def get_interpretation(class_name, confidence):
    """Get medical interpretation"""
    interpretations = {
        'glioma': 'Glioma tumors arise from glial cells in the brain. These are the most common type of primary brain tumor. They can vary in grade from low to high.',
        'meningioma': 'Meningioma tumors develop in the meninges, the protective membranes surrounding the brain and spinal cord. Most are benign (non-cancerous).',
        'pituitary': 'Pituitary tumors originate in the pituitary gland, which controls many body functions through hormone production. Most are benign.',
        'notumor': 'The scan appears normal with no signs of tumor. Regular follow-up imaging may be recommended as part of routine care.'
    }
    return interpretations.get(class_name.lower(), 'Analysis complete.')

# ==================== MAIN APPLICATION ====================
def main():
    # Load model and configuration
    model, config, model_loaded, status_msg = load_model_and_config()
    
    img_size = config.get('img_size', 224) if config else 224
    class_names = config.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary']) if config else []
    
    # Header
    st.markdown("""
        <div class="header-container">
            <h1>🧠 Brain Tumor Detection System</h1>
            <p>CNN + MobileNet Ensemble Learning Model</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check if model is loaded
    if not model_loaded:
        st.error(f"❌ {status_msg}")
        st.info("Make sure ensemble_model_fixed.h5 and config.json are in the same directory as this app.")
        return
    
    # Success message
    st.success(f"✅ {status_msg}")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings & Info")
        
        # Model information
        st.subheader("📊 Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Classes", len(class_names))
        with col2:
            st.metric("Image Size", f"{img_size}×{img_size}")
        
        st.divider()
        
        # Available classes
        st.subheader("🏥 Tumor Classes")
        for i, class_name in enumerate(class_names, 1):
            status = "✓" if class_name == 'notumor' else "⚠"
            st.write(f"{status} {i}. {class_name.upper()}")
        
        st.divider()
        
        # System status
        st.subheader("✅ System Status")
        st.write("Model: Loaded ✓")
        st.write("Config: Loaded ✓")
        st.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.divider()
        
        # About
        st.subheader("ℹ️ About This System")
        st.write("""
        **Architecture:**
        - Custom CNN layers
        - MobileNetV2 transfer learning
        - Ensemble averaging
        
        **Input:** MRI brain scan images
        **Output:** 4-class tumor classification
        **Accuracy:** ~95% on validation set
        """)
        
        st.divider()
        
        st.warning("⚠️ **Disclaimer**: For research & educational purposes only. Consult medical professionals for diagnosis.")
    
    # Initialize session state
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None
    
    # Main content layout
    st.subheader("📤 Upload MRI Scan")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an MRI scan image",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col_img, col_info = st.columns([1, 1]) if st.session_state.show_results else (st.container(), None)
        
        with col_img:
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
            
            # Action buttons
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🔍 Analyze Image", use_container_width=True, key="analyze_btn"):
                    st.session_state.show_results = True
                    st.session_state.uploaded_image = image
                    st.session_state.uploaded_filename = uploaded_file.name
            
            with col_btn2:
                if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
                    st.session_state.show_results = False
                    st.session_state.uploaded_image = None
                    st.rerun()
        
        
        
        # Results section
        if st.session_state.show_results and st.session_state.uploaded_image:
            with col_info if col_info else st.container():
                st.subheader("📊 Analysis Results")
                
                with st.spinner("🔄 Analyzing MRI scan..."):
                    # Make prediction
                    predictions, predicted_idx = predict_tumor(
                        st.session_state.uploaded_image, 
                        model, 
                        img_size
                    )
                    
                    predicted_class = class_names[predicted_idx]
                    confidence = predictions[predicted_idx] * 100
                
                st.success("✅ Analysis Complete")
                
                # Main prediction box
                st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 2rem;
                        border-radius: 15px;
                        text-align: center;
                        margin-bottom: 1.5rem;
                        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                    ">
                        <h3 style="margin: 0; opacity: 0.9;">Predicted Class</h3>
                        <h1 style="margin: 0.5rem 0 0 0; font-size: 2.5rem;">{predicted_class.upper()}</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.metric("Confidence", f"{confidence:.2f}%", delta=f"{confidence:.2f}")
                
                with col_m2:
                    if predicted_class.lower() == 'notumor':
                        st.metric("Status", "✅ Healthy")
                    else:
                        st.metric("Status", "⚠️ Alert")
                
                with col_m3:
                    st.metric("Time", datetime.now().strftime("%H:%M:%S"))
                
                st.divider()
                
                # Status alert
                if predicted_class.lower() == 'notumor':
                    st.markdown("""
                        <div class="success-alert">
                        ✓ No tumor detected - Healthy brain scan
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="danger-alert">
                        ⚠️ Tumor detected: {predicted_class.upper()}
                        </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # Probabilities visualization
                st.subheader("📈 Probability Analysis")
                
                # Bar chart
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=[name.upper() for name in class_names],
                        y=predictions * 100,
                        marker=dict(
                            color=predictions * 100,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Confidence %", thickness=15, len=0.7)
                        ),
                        text=[f"{pred*100:.2f}%" for pred in predictions],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2f}%<extra></extra>'
                    )
                ])
                
                fig_bar.update_layout(
                    title="Confidence Scores by Class",
                    xaxis_title="Tumor Class",
                    yaxis_title="Confidence (%)",
                    height=400,
                    hovermode='x unified',
                    plot_bgcolor='rgba(248, 249, 255, 0.5)',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[name.upper() for name in class_names],
                    values=predictions * 100,
                    textposition='inside',
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>Probability: %{value:.2f}%<extra></extra>',
                    marker=dict(line=dict(color='white', width=2))
                )])
                
                fig_pie.update_layout(
                    title="Probability Distribution",
                    height=400,
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.divider()
                
                # Medical interpretation
                st.subheader("📋 Medical Interpretation")
                interpretation = get_interpretation(predicted_class, confidence)
                st.info(interpretation)
                
                st.divider()
                
                # Detailed table
                st.subheader("📊 Detailed Probability Breakdown")
                
                table_data = []
                for i, name in enumerate(class_names):
                    prob_pct = predictions[i] * 100
                    table_data.append({
                        "Class": name.upper(),
                        "Probability": f"{prob_pct:.2f}%",
                        "Score": f"{predictions[i]:.4f}"
                    })
                
                st.dataframe(table_data, use_container_width=True, hide_index=True)
                
                st.divider()
                
                # Download report
                st.subheader("📥 Download Analysis Report")
                
                report = f"""
================================================================================
                    BRAIN TUMOR DETECTION - ANALYSIS REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {st.session_state.get('uploaded_filename', 'unknown')}

================================================================================
                           MAIN PREDICTION
================================================================================
Predicted Class: {predicted_class.upper()}
Confidence Level: {confidence:.2f}%
Status: {'Healthy (No Tumor)' if predicted_class.lower() == 'notumor' else 'Tumor Detected'}

================================================================================
                       DETAILED PROBABILITIES
================================================================================
"""
                
                for i, name in enumerate(class_names):
                    report += f"{name.upper():15} : {predictions[i]*100:7.2f}%\n"
                
                report += f"""
================================================================================
                         INTERPRETATION
================================================================================
{interpretation}

================================================================================
                          CONFIDENCE SCORES
================================================================================
"""
                
                for i, name in enumerate(class_names):
                    bar = "█" * int(predictions[i] * 50)
                    report += f"{name.upper():15} [{bar:<50}] {predictions[i]*100:.2f}%\n"
                
                report += f"""
================================================================================
                           DISCLAIMER
================================================================================
This system is designed for research and educational purposes only.
It should NOT be used for actual medical diagnosis.
For medical diagnosis and treatment, please consult with qualified medical professionals.

The predictions are based on machine learning algorithms and may have limitations.
Always seek professional medical advice for accurate diagnosis.

================================================================================
"""
                
                st.download_button(
                    label="📄 Download Report (TXT)",
                    data=report,
                    file_name=f"tumor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    else:
        st.info("👆 Upload an MRI scan image to get started")
        st.write("")
        
        # Example information
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            st.subheader("📋 Supported Formats")
            st.write("- PNG (.png)")
            st.write("- JPEG (.jpg, .jpeg)")
            st.write("- GIF (.gif)")
            st.write("- BMP (.bmp)")
        
        with col_ex2:
            st.subheader("📏 Requirements")
            st.write("- File size: < 10 MB")
            st.write("- Dimensions: Any size (will be resized)")
            st.write("- Color: RGB/Grayscale")
            st.write("- Quality: High resolution preferred")

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()
