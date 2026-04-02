import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

import asyncio
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

from digital_empathy.brain_regions import load_brain_masks
from digital_empathy.scoring import compute_friction_score
from digital_empathy.visualization import render_brain_heatmap
from digital_empathy.inference import TribeInferenceEngine

# Page config
st.set_page_config(
    page_title="Digital Empathy — Cognitive Friction Analyzer",
    layout="wide",
    page_icon="🧠"
)

# Cache load_brain_masks()
@st.cache_resource
def get_masks():
    return load_brain_masks()

# Cache TribeInferenceEngine
@st.cache_resource
def get_engine():
    return asyncio.run(TribeInferenceEngine.create(cache_dir="./cache"))

# Header
st.title("Digital Empathy")
st.markdown("### Giving AI agents a window into the human mind")
st.markdown("Upload a screen recording of a human testing a UI. TRIBE v2 predicts their brain activity and returns a Cognitive Friction Score.")
st.divider()

# Sidebar
uploaded_file = st.sidebar.file_uploader("Upload recording", type=["mp4", "mov", "avi", "webm"])
analyze_button = st.sidebar.button("Analyze Recording", disabled=uploaded_file is None)
st.sidebar.info("1. Record a human testing your UI\n2. Upload here\n3. Get a brain-based friction score")

if "engine_loaded" in st.session_state:
    st.sidebar.markdown('🟢 **Model loaded**')
else:
    st.sidebar.markdown('🟡 **Model not yet loaded**')

if analyze_button and uploaded_file is not None:
    if uploaded_file.size > 500 * 1024 * 1024:
        st.warning("File too large — please trim the recording to under 500MB")
    else:
        try:
            # Create uploads dir
            upload_dir = Path("./output/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Save uploaded file
            temp_path = upload_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with st.spinner("Loading models..."):
                masks = get_masks()
                engine = get_engine()
                st.session_state.engine_loaded = True
                
            with st.spinner("Running TRIBE v2 neural prediction…"):
                prediction = asyncio.run(engine.predict_from_video(str(temp_path)))
                
            with st.spinner("Computing cognitive friction score..."):
                result = compute_friction_score(prediction.activations, masks)

            with st.spinner("Rendering brain heatmap…"):
                heatmap_path = render_brain_heatmap(
                    activations=prediction.activations,
                    video_path=str(temp_path),
                    output_dir="./output",
                    pfc_mask=masks.pfc,
                )
                # Cleanup plt to prevent memory accumulation between re-runs
                plt.close("all")

            st.success(f"Analysis complete — score: {result.score:.1f} / 10")
            if result.score <= 3.0:
                st.balloons()
            
            col1, col2 = st.columns([1.1, 1])
            
            with col1:
                # Dark color scheme / CSS
                st.markdown("""
                <style>
                .score-panel {
                    background-color: #1e1e1e;
                    padding: 2rem;
                    border-radius: 10px;
                    margin-bottom: 1rem;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="score-panel">', unsafe_allow_html=True)
                st.markdown(f'<h1 style="font-size: 72px; margin: 0;">{result.score:.1f} / 10</h1>', unsafe_allow_html=True)
                
                color_map = {
                    "Effortless": "#22c55e",
                    "Comfortable": "#84cc16",
                    "Moderate": "#eab308",
                    "Strained": "#f97316",
                    "High Friction": "#ef4444",
                    "Critical": "#7f1d1d"
                }
                bg_color = color_map.get(result.label, "#gray")
                
                st.markdown(f'<span style="background-color: {bg_color}; color: white; padding: 0.5rem 1rem; border-radius: 9999px; font-weight: bold; font-size: 1.2rem;">{result.label}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.progress(result.score / 10.0)
                st.info(result.explanation)
                
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Cognitive Load Ratio", f"{result.cognitive_load_ratio:.3f}")
                sm2.metric("PFC Activation", f"{result.pfc_mean_activation:.4f}")
                sm3.metric("Visual Activation", f"{result.visual_mean_activation:.4f}")
                
            with col2:
                st.image(heatmap_path, caption="Predicted neural activation — orange overlay = Prefrontal Cortex (cognitive load region)")
                with open(heatmap_path, "rb") as f:
                    st.download_button("Download Heatmap", data=f, file_name=Path(heatmap_path).name, mime="image/png")
                    
            st.divider()
            st.subheader("Raw MCP Response Payload")
            st.markdown("This is the exact JSON the AI coding agent receives")
            st.json(result.to_dict())
                
        except Exception as e:
            st.error(f"Error during analysis: {e}")
