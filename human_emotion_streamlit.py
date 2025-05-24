import streamlit as st
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
import joblib
from datetime import datetime
import os
import matplotlib.pyplot as plt

# --- Styling ---
st.set_page_config(page_title="Emotion Detection System", layout="wide", page_icon=":sound:")
st.markdown("""
    <style>
        body {background: linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important;}
        .stApp {background: linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important;}
        .main-card {
            background: rgba(255,255,255,0.15);
            border-radius: 18px;
            padding: 2.5rem 2rem 2rem 2rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.18);
            animation: fadeIn 1.2s;
        }
        .gradient-header {
            background: linear-gradient(90deg,#667eea,#764ba2,#ee9ca7,#ffdde1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: 2px;
            margin-bottom: 0.5rem;
            text-align: center;
            animation: gradientMove 5s linear infinite;
            background-size: 200% 200%;
        }
        .subheader {
            text-align: center;
            color: #fff;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            letter-spacing: 1px;
            opacity: 0.85;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        @keyframes gradientMove {
            0% {background-position: 0% 50%;}
            100% {background-position: 100% 50%;}
        }
        .emotion-indicator {
            border-radius: 12px;
            padding: 1.2rem 0.5rem;
            margin: 0.2rem;
            background: rgba(255,255,255,0.18);
            box-shadow: 0 2px 8px rgba(118,75,162,0.12);
            text-align: center;
            transition: transform 0.25s;
            font-weight: 600;
            font-size: 1.1rem;
        }
        .emotion-indicator:hover {
            transform: translateY(-7px) scale(1.03);
            box-shadow: 0 4px 16px rgba(118,75,162,0.22);
        }
        .confidence {
            font-size: 1.7rem;
            font-weight: 800;
            margin-top: 0.2rem;
            background: linear-gradient(90deg,#667eea,#ee9ca7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .footer {
            text-align: center;
            color: #fff;
            margin-top: 2.5rem;
            font-size: 1rem;
            opacity: 0.7;
        }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
EMOTIONS = {
    'neutral': '😐',
    'calm': '😌',
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fearful': '😨',
    'disgust': '🤢',
    'surprised': '😲'
}
MODEL_PATH = 'emotion_model.pkl'
LOG_FILE = 'session_log.csv'
AUDIO_FILE = 'recorded.wav'

# --- Utility Functions ---
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])

def record_voice(filename=AUDIO_FILE, duration=3, fs=44100):
    st.info("Recording... Please speak into your microphone.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, audio)
    st.success("Recording complete!")

def log_session(emotion):
    log_entry = pd.DataFrame([[datetime.now(), emotion]], columns=["Timestamp", "Emotion"])
    if not os.path.exists(LOG_FILE):
        log_entry.to_csv(LOG_FILE, index=False)
    else:
        log_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)

def load_history():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    else:
        return pd.DataFrame(columns=["Timestamp", "Emotion"])

def plot_emotion_history(df):
    if df.empty:
        return
    emotion_counts = df['Emotion'].value_counts()
    fig, ax = plt.subplots(figsize=(6,2.5))
    colors = plt.cm.plasma(np.linspace(0,1,len(emotion_counts)))
    emotion_counts.plot(kind='bar', color=colors, ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Emotion History")
    st.pyplot(fig)

# --- Main App ---
st.markdown('<div class="gradient-header">Emotion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">AI-Powered Voice Emotion Recognition with Modern UI</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("🎤 Record Your Voice")
        st.write("Click below and speak for 3 seconds. Your emotion will be detected from your voice.")
        if st.button("Start Recording", use_container_width=True):
            record_voice()
            st.balloons()
    with col2:
        st.subheader("🎧 Or Upload a WAV File")
        uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])
        if uploaded_file:
            with open(AUDIO_FILE, "wb") as f:
                f.write(uploaded_file.read())
            st.success("Audio file uploaded!")

    # --- Emotion Prediction ---
    st.markdown("---")
    st.subheader("🔎 Analyze Emotion")
    analyze = st.button("Analyze Emotion", use_container_width=True, type="primary")
    if analyze:
        if not os.path.exists(AUDIO_FILE):
            st.warning("No audio recorded or uploaded yet.")
        else:
            try:
                features = extract_features(AUDIO_FILE).reshape(1, -1)
                model = joblib.load(MODEL_PATH)
                proba = model.predict_proba(features)[0]
                classes = model.classes_
                pred_idx = np.argmax(proba)
                pred_emotion = classes[pred_idx]
                confidence = proba[pred_idx]
                log_session(pred_emotion)

                # --- Show Result ---
                st.markdown("#### Detected Emotion:")
                colA, colB, colC = st.columns(3)
                for i, (emo, emoji) in enumerate(EMOTIONS.items()):
                    idx = np.where(classes==emo)[0]
                    conf = proba[idx[0]] if len(idx) else 0
                    color = "#667eea" if emo == pred_emotion else "#fff"
                    with [colA, colB, colC][i%3]:
                        st.markdown(
                            f"""
                            <div class="emotion-indicator" style="border: 2.5px solid {color};">
                                <div style="font-size:2.3rem;">{emoji}</div>
                                <div>{emo.capitalize()}</div>
                                <div class="confidence">{conf*100:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True
                        )
                st.success(f"**{pred_emotion.capitalize()}** detected with {confidence*100:.1f}% confidence.")

                # --- Show Probability Bar ---
                st.markdown("##### Emotion Probabilities")
                prob_df = pd.DataFrame({"Emotion": classes, "Confidence": proba*100})
                st.bar_chart(data=prob_df.set_index("Emotion"))

            except Exception as e:
                st.error(f"Error analyzing emotion: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

# --- Session History ---
st.markdown("## 📜 Session History")
history_df = load_history()
if not history_df.empty:
    st.dataframe(history_df.tail(10).sort_values("Timestamp", ascending=False), use_container_width=True)
    plot_emotion_history(history_df)
else:
    st.info("No session history yet.")

# --- Footer ---
st.markdown('<div class="footer">Powered by AI Emotion Recognition | &copy; 2025</div>', unsafe_allow_html=True)
