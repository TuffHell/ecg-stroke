import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import butter, filtfilt

# --- 1. DEFINE THE AI ARCHITECTURE (Must match your trained model exactly) ---
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
    # Load the downloaded .pth file
    model.load_state_dict(torch.load("strokewatch_v1.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# --- 3. STREAMLIT INTERFACE ---
st.set_page_config(page_title="StrokeWatch AI", page_icon="🫀", layout="centered")

st.title("🫀 StrokeWatch AI: Atrial Fibrillation Detection")
st.write("Upload a raw ECG signal file (.npy format) to analyze for AFib risk using our PyTorch CNN.")

# Load the AI model
ai_model = load_model()

# File uploader widget
uploaded_file = st.file_uploader("Upload 10-second ECG data (.npy)", type=["npy"])

if uploaded_file is not None:
    # Load the uploaded numpy file
    ecg_data = np.load(uploaded_file)
    
    st.write("✅ **File loaded successfully. Analyzing...**")
    
    # Pre-process the exact same way as training
    # Assuming shape is [time_steps, channels] -> extract lead 0
    if len(ecg_data.shape) > 1:
        signal = ecg_data[:, 0:1] # Take Lead I
    else:
        signal = ecg_data.reshape(-1, 1) # Fallback if 1D array
        
    signal_filtered = apply_bandpass_filter(np.expand_dims(signal, axis=0))
    
    # Format for PyTorch: [Batch, Channels, Time]
    tensor_signal = torch.tensor(np.transpose(signal_filtered[0]), dtype=torch.float32).unsqueeze(0)
    
    # Make the prediction
    with torch.no_grad():
        raw_score = ai_model(tensor_signal)
        risk_prob = torch.sigmoid(raw_score).item()
        
    # Display the results professionally
    st.markdown("---")
    st.subheader("Diagnostic Results")
    
    threshold = 0.45
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="AI Confidence Score", value=f"{risk_prob * 100:.1f}%")
    with col2:
        if risk_prob > threshold:
            st.error("⚠️ **AFIB DETECTED**")
            st.write("Irregular rhythm matching Atrial Fibrillation morphology.")
        else:
            st.success("✅ **NORMAL RHYTHM**")
            st.write("No distinct markers of AFib detected.")
            
    st.caption("Disclaimer: StrokeWatch is an AI prototype. Always consult a physician for medical advice.")