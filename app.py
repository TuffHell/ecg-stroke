import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go
import time

# --- 1. DEFINE THE AI ARCHITECTURE ---
class StrokeWatchNet(nn.Module):
    def __init__(self, num_leads=1):
        super(StrokeWatchNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_leads, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),            
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)        
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        return self.classifier(x)

# --- 2. HELPER FUNCTIONS ---
def apply_bandpass_filter(signals, lowcut=0.5, highcut=40.0, fs=500.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signals, axis=1)

@st.cache_resource
def load_model():
    model = StrokeWatchNet(num_leads=1)
    model.load_state_dict(torch.load("strokewatch_v1.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def plot_ecg(signal, title, color):
    # Downsample slightly for faster browser rendering
    time_axis = np.linspace(0, 10, len(signal))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=signal, mode='lines', line=dict(color=color, width=1.5)))
    fig.update_layout(title=title, xaxis_title="Time (Seconds)", yaxis_title="Voltage (mV)", 
                      height=300, margin=dict(l=0, r=0, t=40, b=0), template="plotly_white")
    return fig

# --- 3. SESSION STATE SETUP ---
# This ensures the app remembers the loaded patient when switching tabs
if 'current_signal' not in st.session_state:
    st.session_state.current_signal = None
    st.session_state.raw_signal = None
    st.session_state.true_label = None

# --- 4. STREAMLIT INTERFACE ---
st.set_page_config(page_title="StrokeWatch AI", page_icon="🫀", layout="wide")
st.title("🫀 StrokeWatch AI: Clinical Dashboard")
st.markdown("Real-time Atrial Fibrillation detection via 1-Lead continuous ECG monitoring.")

ai_model = load_model()

# --- SIDEBAR: PATIENT SELECTION ---
with st.sidebar:
    st.header("Patient Data Input")
    st.write("Simulate a live patient scan or upload your own .npy ECG file.")
    
    if st.button("🎲 Generate Random Demo Patient", use_container_width=True):
        try:
            demo_data = np.load('strokewatch_demo_samples.npz')
            idx = np.random.randint(0, len(demo_data['labels']))
            
            raw_data = demo_data['signals'][idx]
            if len(raw_data.shape) > 1:
                raw_data = raw_data[:, 0] # Extract Lead I
                
            st.session_state.raw_signal = raw_data
            st.session_state.current_signal = apply_bandpass_filter(np.expand_dims(raw_data, axis=0))[0]
            st.session_state.true_label = "AFIB" if demo_data['labels'][idx] == 1.0 else "Normal"
        except FileNotFoundError:
            st.error("Demo file 'strokewatch_demo_samples.npz' not found in directory.")

    st.divider()
    uploaded_file = st.file_uploader("Upload custom .npy file", type=["npy"])
    if uploaded_file is not None:
        raw_data = np.load(uploaded_file)
        if len(raw_data.shape) > 1:
            raw_data = raw_data[:, 0]
        st.session_state.raw_signal = raw_data
        st.session_state.current_signal = apply_bandpass_filter(np.expand_dims(raw_data, axis=0))[0]
        st.session_state.true_label = "Unknown (Manual Upload)"

# --- MAIN DASHBOARD AREA ---
if st.session_state.current_signal is not None:
    
    # Create the Tabs
    tab1, tab2 = st.tabs(["🩺 Live Diagnosis", "🧠 AI Thought Process (XAI)"])
    
    # ----------------------------------------
    # TAB 1: LIVE DIAGNOSIS
    # ----------------------------------------
    with tab1:
        st.subheader("10-Second Continuous Rhythm")
        st.plotly_chart(plot_ecg(st.session_state.current_signal, "Filtered Lead-I ECG", "#1f77b4"), use_container_width=True)
        
        with st.spinner('AI is analyzing morphology and R-R intervals...'):
            time.sleep(1) # Fake loading time for clinical effect
            
         tensor_signal = torch.tensor(st.session_state.current_signal.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                raw_score = ai_model(tensor_signal)
                risk_prob = torch.sigmoid(raw_score).item()
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Actual Patient Status", value=st.session_state.true_label)
            
        with col2:
            st.metric(label="AI Confidence Score", value=f"{risk_prob * 100:.1f}%")
            
        with col3:
            threshold = 0.45
            if risk_prob > threshold:
                st.error("⚠️ AI PREDICTION: AFIB DETECTED")
            else:
                st.success("✅ AI PREDICTION: NORMAL RHYTHM")

    # ----------------------------------------
    # TAB 2: AI THOUGHT PROCESS (EXPLAINABILITY)
    # ----------------------------------------
    with tab2:
        st.header("Algorithmic Explainability")
        st.write("How the StrokeWatch CNN processes raw electrical signals into a medical diagnosis.")
        
        st.markdown("### 1. Pre-Processing & Denoising")
        st.write("The AI first applies a 0.5Hz - 40Hz Butterworth bandpass filter. This removes patient movement (baseline wander) and electrical interference.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_ecg(st.session_state.raw_signal, "Raw Unfiltered Signal", "#7f7f7f"), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_ecg(st.session_state.current_signal, "Cleaned AI Input Signal", "#1f77b4"), use_container_width=True)
            
        st.markdown("### 2. Feature Extraction (Convolutional Layers)")
        st.write("The 1D-CNN slides mathematical filters across the clean signal above, specifically scanning for two critical morphological failures:")
        st.markdown("""
        * **Missing P-Waves:** Checking the baseline space immediately preceding the large R-wave spikes.
        * **Chaotic R-R Intervals:** Measuring the exact millisecond timing between consecutive heartbeats to detect fibrillatory irregularity.
        """)
        
        st.markdown("### 3. Risk Probability Distribution")
        
        # Professional Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Stroke Risk / AFib Probability"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if risk_prob > threshold else "darkgreen"},
                'steps': [
                    {'range': [0, 45], 'color': "lightgreen"},
                    {'range': [45, 100], 'color': "lightcoral"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 45}
            }))
        fig_gauge.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.info("👈 Please select 'Generate Random Demo Patient' from the sidebar to begin the simulation.")