import streamlit as st
import tempfile
import os
import time
from brain_core import CIPipelineValidator

st.set_page_config(page_title="CI Risk Validator", page_icon="🧠", layout="centered")

st.title("🧠 CI Pipeline Risk Validator")
st.write("Upload a CI pipeline JSON file to analyze risk before execution.")

@st.cache_resource
def load_validator():
    return CIPipelineValidator()

try:
    validator = load_validator()
except Exception as e:
    st.error(f"Error loading models. Make sure 'models' folder exists: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload pipeline JSON", type=["json"])

if uploaded_file is not None:
    if st.button("Analyze Pipeline"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
                
            status_text.text("Starting analysis...")
            progress_bar.progress(10)
            time.sleep(0.3)
            
            status_text.text("Validating syntax and schema...")
            progress_bar.progress(30)
            time.sleep(0.3)
            
            status_text.text("Extracting features...")
            progress_bar.progress(55)
            time.sleep(0.3)
            
            status_text.text("Running ML risk prediction...")
            progress_bar.progress(75)
            
            result = validator.validate_file_dict(tmp_path)
            os.remove(tmp_path)
            
            status_text.text("Finalizing results...")
            progress_bar.progress(90)
            time.sleep(0.2)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete ✅")
            
            risk_level = result.get("risk_level", "UNKNOWN")
            score = result.get("final_risk_score", 0)
            
            if risk_level == "LOW":
                st.success(f"✅ LOW RISK ({score} / 100)")
            elif risk_level == "MEDIUM":
                st.warning(f"⚠️ MEDIUM RISK ({score} / 100)")
            else:
                st.error(f"🚨 HIGH RISK ({score} / 100)")
                
            st.subheader("Detailed Report")
            st.json(result)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Analysis failed: {e}")
