import streamlit as st
import base64
import os
import sys

# Add src to pythonpath so it works seamlessly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference.workflow_preprocessor import WorkflowPreprocessor
from src.inference.explainer import generate_explanation

st.set_page_config(page_title="RunSure - CI/CD Risk Classifier", page_icon="⚙️", layout="centered")

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_path = os.path.join("frontend", "public", "assets", "starry_night.jpg")
b64_bg = get_base64_image(bg_path)

# Custom CSS for the Vincent van Gogh UI with readable contrast
st.markdown(f'''
<style>
/* Background Image and Overlays */
.stApp {{
    background-image: url("data:image/jpg;base64,{b64_bg}");
    background-size: cover;
    background-position: center center;
    background-attachment: fixed;
}}
.stApp::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(5, 10, 20, 0.40); /* Light overlay to retain painting visibility */
    pointer-events: none;
    z-index: -1;
}}

/* Typography & Global Colors */
html, body, [class*="css"] {{
    color: #FFF7E6 !important;
}}
h1, h2, h3, h4, h5, h6 {{
    color: #FFD166 !important;
    text-shadow: 0px 2px 4px rgba(0,0,0,0.8);
}}
p, span, div, label {{
    color: #E8DCC8 !important;
}}

/* Glass Panels with Dark Opaque Backing for Contrast */
.glass-panel {{
    background: rgba(8, 15, 28, 0.84) !important;
    border: 1px solid rgba(255, 209, 102, 0.30) !important;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(6px);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}}

.glass-panel h2, .glass-panel h3 {{
    margin-top: 0;
}}

/* Metrics */
.metric-box {{
    background: rgba(5, 10, 20, 0.90);
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 209, 102, 0.30);
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}}
.metric-value {{
    font-size: 2.2rem;
    font-weight: bold;
    color: #FFD166 !important;
    text-shadow: 0px 2px 4px rgba(0,0,0,0.8);
}}
.metric-label {{
    font-size: 0.9rem;
    color: #C8C0B2 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* Risk Indicator Borders */
.risk-LOW {{ border-left: 5px solid #4ADE80 !important; }}
.risk-MEDIUM {{ border-left: 5px solid #FBBF24 !important; }}
.risk-HIGH {{ border-left: 5px solid #FB7185 !important; }}

/* Streamlit Native Components overrides */
div[data-testid="stFileUploader"] {{
    background: rgba(8, 15, 28, 0.84);
    border: 1px dashed rgba(255, 209, 102, 0.50);
    padding: 1.5rem;
    border-radius: 10px;
}}
div[data-testid="stFileUploader"] > div > div > div > small {{
    color: #C8C0B2 !important;
}}
button[kind="secondary"], button[kind="primary"] {{
    background: rgba(8, 15, 28, 0.9) !important;
    border: 1px solid rgba(255, 209, 102, 0.5) !important;
    color: #FFD166 !important;
    font-weight: 600;
    transition: all 0.3s ease;
}}
button[kind="secondary"]:hover, button[kind="primary"]:hover {{
    background: rgba(255, 209, 102, 0.15) !important;
    border-color: #FFD166 !important;
    color: #FFF !important;
}}
div[data-testid="stExpander"] {{
    background: rgba(8, 15, 28, 0.84);
    border: 1px solid rgba(255, 209, 102, 0.30);
    border-radius: 8px;
}}
div[data-testid="stExpander"] summary {{
    color: #FFD166 !important;
}}

/* Success/Error/Warning Banners */
div[data-testid="stAlert"] {{
    background: rgba(8, 15, 28, 0.9) !important;
    border: 1px solid rgba(255, 209, 102, 0.4);
    color: #FFF7E6 !important;
}}

</style>
''', unsafe_allow_html=True)

def get_prediction_display_data(prediction):
    if not isinstance(prediction, dict):
        raise ValueError("Prediction must be a dictionary.")
        
    pred_class = prediction.get("predicted_class")
    pred_label = prediction.get("predicted_label")
    probs = prediction.get("probabilities", {})
    
    if not pred_label:
        raise ValueError("Missing 'predicted_label' in prediction output.")
        
    confidence = 0.0
    if isinstance(probs, dict):
        if pred_label in probs:
            confidence = probs[pred_label]
        elif pred_class in probs:
            confidence = probs[pred_class]
        elif str(pred_class) in probs:
            confidence = probs[str(pred_class)]
        else:
            raise ValueError(f"Cannot extract confidence for label '{{pred_label}}' from dict: {{probs}}")
    elif isinstance(probs, list) and isinstance(pred_class, int) and pred_class < len(probs):
        confidence = probs[pred_class]
    else:
        raise ValueError(f"Cannot extract confidence for label '{{pred_label}}' from format: {{probs}}")
        
    return {{
        "class_index": pred_class,
        "label": pred_label,
        "confidence": confidence,
        "probabilities": probs
    }}


@st.cache_resource
def load_preprocessor():
    try:
        prep = WorkflowPreprocessor()
        # Initialize the model eagerly to avoid cold start on first request
        prep.predict("name: dummy\non: push\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo")
        return prep
    except Exception as e:
        st.error(f"Failed to load model backend: {{e}}")
        return None

prep = load_preprocessor()

# Wrap main content in a glass panel for superior contrast against the background
st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
st.markdown("<h1>⚙️ RunSure - CI/CD Workflow Risk Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a GitHub Actions workflow YAML to classify its pre-execution risk.</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload GitHub Actions Workflow", type=["yml", "yaml"])

if uploaded_file is not None and prep is not None:
    if st.button("Analyze Workflow"):
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            
            with st.spinner("Analyzing Workflow via E06 Hybrid Logistic Regression..."):
                result = prep.predict(content)
            
            # Robust extraction using helper
            display_data = get_prediction_display_data(result["prediction"])
            
            meta = result["preprocessing"]
            latency = result["inference_duration"] * 1000.0
            risk_label = display_data["label"]
            confidence = display_data["confidence"]
            
            # Generate explanation
            explanation = None
            try:
                explanation = generate_explanation(risk_label, meta["structural_features"])
            except Exception:
                pass

            color = "#4ADE80" if risk_label == "LOW" else "#FBBF24" if risk_label == "MEDIUM" else "#FB7185"
            
            st.markdown(f'''
            <div class="glass-panel risk-{risk_label}">
                <h2 style="color: {color} !important; text-shadow: none;">{risk_label} RISK DETECTED</h2>
                <p style="font-size: 1.1rem; color: #FFF7E6 !important;">Confidence: {confidence * 100:.1f}%</p>
            </div>
            ''', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-value">{meta.get("job_count", 0)}</div>
                    <div class="metric-label">Jobs</div>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-value">{meta.get("step_count", 0)}</div>
                    <div class="metric-label">Steps</div>
                </div>
                ''', unsafe_allow_html=True)

            st.markdown(f"<p style='text-align: right; font-size: 0.85rem; color: #C8C0B2 !important; margin-top: 10px;'>Inference Latency: {latency:.1f} ms</p>", unsafe_allow_html=True)
            
            if explanation and explanation.get("signals"):
                st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: #FFD166 !important;'>Why this risk?</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #FFF7E6 !important; font-weight: 600;'>{explanation['title']}</p>", unsafe_allow_html=True)
                for sig in explanation["signals"]:
                    st.markdown(f'''
                    <div style="background: rgba(5, 10, 20, 0.6); padding: 1rem; margin-bottom: 0.75rem; border-left: 3px solid {color} !important; border-radius: 6px;">
                        <strong style="color: #FFF7E6 !important; font-size: 1.05rem;">{sig['name']}</strong><br/>
                        <span style="font-size: 0.95rem; color: #E8DCC8 !important;">{sig['detail']}</span>
                    </div>
                    ''', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Model Diagnostics & Raw Probabilities"):
                st.json(result)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.exception(e)
