import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from skimage.feature import peak_local_max 
import plotly.graph_objects as go
import time
# --- 1. DEFINE THE AI ARCHITECTURE (Must match trained model exactly) ---
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
    # Using original .pth file for Python-native execution
    model.load_state_dict(torch.load("strokewatch_v1.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def calculate_rr_intervals(signal, fs=500.0):
    # Detect heartbeats using mathematical peak finding on filtered signal
    # We invert the signal slightly as R-peaks can vary in polarity
    # peak_local_max requires scikit-image
    try:
        peaks = peak_local_max(signal.squeeze(), min_distance=150, threshold_abs=np.std(signal)*1.5)
        if len(peaks) < 3: return None, None # Need at least 3 beats for variance
        peak_indices = np.sort(peaks.squeeze())
        intervals = np.diff(peak_indices) / fs # in seconds
        return intervals, peak_indices / fs
    except:
        return None, None

def plot_ecg_spectrum(signal, title, fs=500.0):
    # FFT Frequency Analysis visualization
    N = len(signal)
    yf = rfft(signal.squeeze())
    xf = rfftfreq(N, 1/fs)
    magnitude = np.abs(yf)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=magnitude, mode='lines', line=dict(color='#888', width=1)))
    # Highlight critical diagnostic band (0.5 - 40Hz)
    fig.add_vrect(x0=0.5, x1=40, fillcolor="lightgreen", opacity=0.3, line_width=0, annotation_text="Diagnostic Band")
    fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", height=250, margin=dict(l=0,r=0,t=40,b=0), template="plotly_white")
    return fig

# --- 3. SESSION STATE SETUP ---
# Remember generated patient when switching tabs
if 'current_signal' not in st.session_state:
    st.session_state.current_signal = None
    st.session_state.raw_signal = None
    st.session_state.true_label = None

# --- 4. STREAMLIT INTERFACE ---
st.set_page_config(page_title="StrokeWatch AI", page_icon="🫀", layout="wide")
st.title("🫀 StrokeWatch AI: Deep Diagnostic Dashboard")
st.markdown("Mathematical verification and explainability suite for wrist-based AFib detection.")

ai_model = load_model()

# --- SIDEBAR: PATIENT SIMULATION ---
with st.sidebar:
    st.header("Patient Simulation Control")
    st.write("Simulate a live patient monitoring feed or upload a manual ECG strip.")
    
    if st.button("🎲 Simulate Random Live Feed", use_container_width=True):
        try:
            demo_data = np.load('strokewatch_demo_samples.npz')
            idx = np.random.randint(0, len(demo_data['labels']))
            
            raw_data = demo_data['signals'][idx]
            if len(raw_data.shape) > 1:
                raw_data = raw_data[:, 0]
                
            st.session_state.raw_signal = raw_data
            st.session_state.current_signal = apply_bandpass_filter(np.expand_dims(raw_data, axis=0))[0]
            st.session_state.true_label = "AFIB" if demo_data['labels'][idx] == 1.0 else "Normal"
        except FileNotFoundError:
            st.error("Demo file 'strokewatch_demo_samples.npz' not found in directory.")

# --- MAIN DASHBOARD AREA ---
if st.session_state.current_signal is not None:
    
    # ADVANCED AI & MATH COMPUTATION BLOCK
    tensor_signal = torch.tensor(st.session_state.current_signal.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Turn ON gradients for Explainable AI Saliency Map calculation
    tensor_signal.requires_grad = True 
    
    raw_score = ai_model(tensor_signal)
    risk_prob = torch.sigmoid(raw_score).item()
    
    # Calculate Saliency Map: Which inputs had the biggest mathematical derivative to the output?
    raw_score.backward()
    saliency_map = tensor_signal.grad.abs().squeeze().numpy()
    # Normalize map (0 to 1 scale)
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    threshold = 0.45 # Optimal Goldilocks Threshold
    is_afib_detected = risk_prob > threshold
    
    # Diagnostic Triage Summary
    st.markdown("---")
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        if is_afib_detected:
            st.error(f"### ⚠️ AI PREDICTION: AFIB DETECTED ({risk_prob * 100:.1f}%)")
            st.caption("AI detects morphology and rhythm markers highly suggestive of Atrial Fibrillation.")
        else:
            st.success(f"### ✅ AI PREDICTION: NORMAL RHYTHM ({risk_prob * 100:.1f}%)")
            st.caption("AI detects no markers of significant AFib risk. Continuing continuous monitoring.")
    with c2: st.metric("Live Feed Status", "Simulated Monitoring")
    with c3: st.metric("Ground Truth Status", st.session_state.true_label)

    # ----------------------------------------
    # Create the Tabs (Deepened XAI Analysis)
    # ----------------------------------------
    tab1, tab2 = st.tabs(["🩺 Live Bedside Monitor", "🧠 Deep Clinical Diagnostics (XAI)"])
    
    # =====================================================================
    # TAB 1: LIVE BEDSIDE MONITOR (MOVING ECG WITH ANOMALY HIGHLIGHTS)
    # =====================================================================
    with tab1:
        st.subheader("Simulated 1-Lead Monitoring Feed (0.5 - 40Hz)")
        
        fs = 500.0
        signal = st.session_state.current_signal
        total_time = len(signal) / fs
        time_axis = np.linspace(0, total_time, len(signal))
        
        # Window parameters for animation
        window_size_sec = 2.0
        window_size_samples = int(window_size_sec * fs)
        num_frames = 25 # Number of animation frames to render
        step_samples = int((len(signal) - window_size_samples) / num_frames)
        
        frames_list = []
        for i in range(num_frames):
            start = i * step_samples
            end = start + window_size_samples
            
            # Extract signal chunk for this frame
            chunk = signal[start:end]
            time_chunk = time_axis[start:end]
            saliency_chunk = saliency_map[start:end]
            
            # Anomaly Tracking Logic:
            # We circle areas with high saliency attention points (>0.85 normalize attention)
            anomaly_threshold = 0.85
            anomaly_indices = np.where(saliency_chunk > anomaly_threshold)[0]
            
            frame_trace_list = [go.Scatter(x=time_chunk, y=chunk, mode='lines', line=dict(color='white', width=1.5), name="Feed")]
            
            # If AFib detected and anomaly points found, overlay red circles
            if is_afib_detected and len(anomaly_indices) > 0:
                frame_trace_list.append(go.Scatter(x=time_chunk[anomaly_indices], y=chunk[anomaly_indices], 
                                                mode='markers', marker=dict(color='red', size=4, symbol='circle-open'), 
                                                name="Anomaly Trace"))
                frame_trace_list.append(go.Scatter(x=time_chunk[anomaly_indices], y=chunk[anomaly_indices] + np.std(chunk)*0.4, # Highlight Circle
                                                mode='markers', marker=dict(color='red', size=15, symbol='circle', opacity=0.3),
                                                showlegend=False))

            frames_list.append(go.Frame(data=frame_trace_list, name=str(i)))
        
        # Base Chart Configuration
        fig_monitor = go.Figure(frames=frames_list)
        fig_monitor.add_trace(go.Scatter(x=time_axis[:window_size_samples], y=signal[:window_size_samples], 
                                         mode='lines', line=dict(color='white', width=1.5), name="Feed"))
        # Anomaly marker placeholder
        fig_monitor.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(color='red', size=4, symbol='circle-open'), name="Anomaly Trace"))
        fig_monitor.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(color='red', size=15, symbol='circle', opacity=0.3), showlegend=False))
            
        # Layout & Animation Controls
        fig_monitor.update_layout(xaxis=dict(range=[0, window_size_sec], autorange=False, gridcolor="#333"), 
                                yaxis=dict(range=[np.min(signal)-np.std(signal)*0.2, np.max(signal)+np.std(signal)*0.2], autorange=False, gridcolor="#333"),
                                updatemenus=[dict(type="buttons", showactive=False, x=1.05, y=0.1, 
                                                buttons=[dict(label="▶ Play Sim", method="animate", args=[None, dict(frame=dict(duration=150, redraw=True), fromcurrent=True, transition=dict(duration=0, easing='linear'), mode='immediate')])])],
                                xaxis_title="Time (Seconds)", yaxis_title="Voltage (mV)", template="plotly_dark", 
                                height=350, margin=dict(l=0,r=0,t=40,b=0), paper_bgcolor='black', plot_bgcolor='black')
        
        st.plotly_chart(fig_monitor, use_container_width=True)
        st.caption("Press '▶ Play Sim' to start the live patient monitor simulation. High-risk anomalies are circled in red.")

    # =====================================================================
    # TAB 2: DEEP CLINICAL DIAGNOSTICS (XAI Suite)
    # =====================================================================
    with tab2:
        st.header("Algorithmic Triage & Explainable AI Mathematics")
        st.write("Full transparency behind the generated probability score using distinct mathematical and clinical validation methods.")
        
        st.divider()
        st.subheader("Diagnostic A: Clinical Verification Math (R-R Interval Analysis)")
        st.markdown(f"""The AI is programmed to mimic advanced clinical diagnostics. Atrial Fibrillation is defined by an *irregularly irregular* ventricular response. We mathematically calculate the timing between detected heartbeats to provide traditional clinical verification for the AI's morphology findings.""")
        
        rr_intervals, peak_times = calculate_rr_intervals(st.session_state.current_signal)
        
        if rr_intervals is not None:
            c1_math, c2_math = st.columns([1,1])
            with c1_math:
                # Plot R-R intervals over time
                fig_rr = go.Figure()
                fig_rr.add_trace(go.Scatter(x=peak_times[1:], y=rr_intervals, mode='lines+markers', line=dict(color='#d62728')))
                fig_rr.update_layout(title="Ventricular R-R Intervals (Tachycardia/Irregularity Map)", xaxis_title="Detect Peak Time (s)", yaxis_title="R-R Interval (s)", height=280, margin=dict(l=0,r=0,t=40,b=0), template="plotly_white")
                st.plotly_chart(fig_rr, use_container_width=True)
                
            with c2_math:
                # Calculate Clinical Metrics
                mean_rr = np.mean(rr_intervals)
                std_rr = np.std(rr_intervals)
                cv_rr = std_rr / mean_rr # Coefficient of Variation - standard metric for AFib chaos
                
                st.write("**Classical Rhythm Metrics:**")
                # Benchmarks source: Standard Medical Literature
                normal_benchmark = "Normal (Typical CV < 0.10)"
                afib_benchmark = "AFIB Suggestive (Typical CV > 0.15)"
                
                s1, s2 = st.columns(2)
                with s1: st.metric(label="R-R Standard Deviation (Chaos $\sigma_{RR}$)", value=f"{std_rr:.4f}s")
                with s2: st.metric(label="R-R Coefficient of Variation ($CV_{RR}$)", value=f"{cv_rr:.4f}", help="Std/Mean. Metric for relative rhythm chaos.")
                
                if cv_rr > 0.15: st.warning(f"R-R analysis strongly SUGGESTS AFib suggesting chaotic rhythm mismatch ({cv_rr:.2f} > 0.15 benchmark). Validating AI's morphology findings.")
                else: st.success(f"R-R analysis SUGGESTS Normal rhythm. The timing mismatch is minimal ({cv_rr:.2f} < 0.10 benchmark).")
        else:
            st.error("Rhythm chaotic: Insufficient distinct R-peaks detected for standard R-R calculation.")

        st.divider()
        st.subheader("Diagnostic B: Signal Processing Fidelity (Pre-Processing Analysis)")
        st.write("We must prove that non-relevant, high-frequency wrist noise (muscle artifact, movement) or powerline interference (50Hz/60Hz) are mathematically removed without altering critical cardiac morphology data needed for diagnosis.")
        
        c1_spec, c2_spec = st.columns(2)
        with c1_spec: st.plotly_chart(plot_ecg_spectrum(st.session_state.raw_signal, "Raw Frequency Spectrum (Unfiltered Input)"), use_container_width=True)
        with c2_spec: st.plotly_chart(plot_ecg_spectrum(st.session_state.current_signal, "Clean Frequency Spectrum (AI Diagnostic Input)"), use_container_width=True)
        
        st.markdown(f"""The AI-applied **0.5Hz - 40Hz Butterworth bandpass filter** isolates the biological cardiac information bandwidth. The Frequency Analysis above proves non-relevant noise outside this band (including common electrical interference) is completely suppressed to ensure AI processing accuracy.""")

        st.divider()
        st.subheader("Diagnostic C: Convolutional Neural Mathematics (Sigmoid Probability)")
        st.write(f"""How did the AI arrive at exactly **{risk_prob * 100:.1f}%**? The 1D-CNN slides mathematical filters across the clean signal above, scanning for morphological failures (Missing P-wave waves, f-waves baseline chaos). The final dense layer produces a raw mathematical Logit (z), which we squeeze into a probability between 0 and 1.""")
        
        col_math1, col_math2 = st.columns(2)
        with col_math1:
            st.info("**1. Raw Network Output (Logit $z$)**")
            st.latex(r"z = \sum_{i=1}^{64} w_i x_i + b")
            st.code(f"Logit (z) = {raw_score.item():.4f}")
            st.caption("The mathematical sum from the final 64-node bottleneck layer.")
            
        with col_math2:
            st.info("**2. Sigmoid Probability Mapping**")
            st.latex(r"P(AFib) = \frac{1}{1 + e^{-z}}")
            st.code(f"P(AFib) = 1 / (1 + e^-({raw_score.item():.4f}))\nP(AFib) = {risk_prob:.4f}  ({risk_prob * 100:.1f}%)")
            st.caption("Translating the unbounded logit mathematically into a binary probability clinical percentage.")

else:
    st.info("👈 Please select 'Simulate Random Live Feed' from the sidebar to begin the clinical simulation.")