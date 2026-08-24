import streamlit as st
import tempfile
import os
import yaml
import json
import base64
from pathlib import Path

# Add src to pythonpath so it works seamlessly
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference.workflow_preprocessor import WorkflowPreprocessor
from src.inference.explainer import generate_explanation

st.set_page_config(page_title="RunSure - CI/CD Risk Classifier", page_icon="🧠", layout="centered")

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_path = os.path.join("frontend", "public", "assets", "starry_night.jpg")
b64_bg = get_base64_image(bg_path)

# Custom CSS for the Vincent van Gogh UI
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
    background: rgba(5, 10, 25, 0.75);
    backdrop-filter: blur(2px);
    z-index: -1;
}}

/* Typography & Colors */
h1, h2, h3, h4, p, span, div {{
    color: #F5F7FA !important;
}}
.glass-panel {{
    background: rgba(15, 22, 38, 0.72) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(8px);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}}
.metric-box {{
    background: rgba(0,0,0,0.3);
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    text-align: center;
}}
.metric-value {{
    font-size: 2rem;
    font-weight: bold;
    color: #6EA8FF !important;
}}
.metric-label {{
    font-size: 0.8rem;
    color: #AEB8C7 !important;
}}
.risk-LOW {{ border-left: 4px solid #4ADE80; }}
.risk-MEDIUM {{ border-left: 4px solid #FBBF24; }}
.risk-HIGH {{ border-left: 4px solid #FB7185; }}

</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_preprocessor():
    try:
        prep = WorkflowPreprocessor()
        # Initialize the model eagerly to avoid cold start on first request
        prep.predict("name: dummy\non: push\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo")
        return prep
    except Exception as e:
        st.error(f"Failed to load model backend: {e}")
        return None

prep = load_preprocessor()

st.markdown("<h1>🧠 RunSure - CI/CD Workflow Risk Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #AEB8C7 !important;'>Upload a GitHub Actions workflow YAML to classify its pre-execution risk.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload YAML Workflow", type=["yml", "yaml", "json"])

if uploaded_file is not None and prep is not None:
    if st.button("Analyze Workflow"):
        try:
            content = uploaded_file.getvalue().decode("utf-8")
            
            # If JSON, we just extract it if it's already parsed
            # But the backend expects YAML string, so we'll just pass the content
            # Wait, if it's json, we convert back to yaml so the parser can handle it?
            # Actually, prep.parse_yaml() uses yaml.safe_load which handles JSON fine!
            
            with st.spinner("Analyzing Workflow via E06 Hybrid Logistic Regression..."):
                result = prep.predict(content)
            
            # Extract Results
            pred = result["prediction"]
            meta = result["preprocessing"]
            latency = result["inference_duration"] * 1000.0
            risk_label = pred["predicted_label"]
            
            # Generate explanation
            explanation = None
            try:
                explanation = generate_explanation(risk_label, meta["structural_features"])
            except Exception:
                pass

            color = "#4ADE80" if risk_label == "LOW" else "#FBBF24" if risk_label == "MEDIUM" else "#FB7185"
            
            st.markdown(f'''
            <div class="glass-panel risk-{risk_label}">
                <h2 style="color: {color} !important;">{risk_label} RISK DETECTED</h2>
                <p>Confidence: {pred["probabilities"][pred["predicted_class"]]*100:.1f}%</p>
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

            st.markdown(f"<p style='text-align: right; font-size: 0.8rem;'>Inference Latency: {latency:.1f} ms</p>", unsafe_allow_html=True)
            
            if explanation and explanation.get("signals"):
                st.markdown("### Why this risk?")
                st.markdown(f"**{explanation['title']}**")
                for sig in explanation["signals"]:
                    st.markdown(f'''
                    <div class="glass-panel" style="padding: 1rem; margin-bottom: 0.5rem; border-left: 3px solid {color} !important;">
                        <strong>{sig['name']}</strong><br/>
                        <span style="font-size: 0.9rem; color: #AEB8C7 !important;">{sig['detail']}</span>
                    </div>
                    ''', unsafe_allow_html=True)

            with st.expander("Model Diagnostics & Raw Probabilities"):
                st.json(result)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.exception(e)
