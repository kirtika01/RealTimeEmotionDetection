import pandas as pd
import numpy as np
import sys
import librosa
from IPython.display import Audio
import keras
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def extract_features(y, sr):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    return {
        "chroma_stft_mean": np.mean(chroma_stft),
        "chroma_stft_std": np.std(chroma_stft),
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_centroid_std": np.std(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "spectral_bandwidth_std": np.std(spectral_bandwidth),
        "rolloff_mean": np.mean(rolloff),
        "rolloff_std": np.std(rolloff),
        "zero_crossing_rate_mean": np.mean(zero_crossing_rate),
        "zero_crossing_rate_std": np.std(zero_crossing_rate),
        **{f"mfcc{i}_mean": np.mean(mfcc[i]) for i in range(mfcc.shape[0])},
        **{f"mfcc{i}_std": np.std(mfcc[i]) for i in range(mfcc.shape[0])},
    }


from joblib import dump, load

rf_model = load(r"C:\Users\hp\Desktop\happy\finalized_model.sav.joblib")
import pyaudio

# --- PyAudio Configuration ---
CHUNK_SIZE = 1024  # Adjust based on your system and latency needs
FORMAT = pyaudio.paInt16
CHANNELS = 1  # If using a mono microphone
RATE = 48000  # Or adjust to match your microphone's sampling rate
# --- Initialize Audio Stream ---
p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
)

# --- Circular Buffer (Simplified for Example) ---
buffer = []
BUFFER_LENGTH = 40  # Adjust this to control analysis length

# --- Main Loop ---
try:
    while True:
        data = stream.read(CHUNK_SIZE)
        buffer.append(data)

        if len(buffer) == BUFFER_LENGTH:
            # Extract audio data from the buffer
            frames = b"".join(buffer)
            y = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

            # Extract features
            features = extract_features(y, RATE)
            df = pd.DataFrame(features, index=[0])

            # Make prediction
            prediction = rf_model.predict(df)[0]
            print("Predicted emotion:", prediction)

            buffer = []  # Clear the buffer

except KeyboardInterrupt:
    print("Exiting...")
    stream.stop_stream()
    stream.close()
    p.terminate()

    import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import pyaudio

# --- PyAudio Configuration ---
CHUNK_SIZE = 1024  # Adjust based on your system and latency needs
FORMAT = pyaudio.paInt16
CHANNELS = 1  # If using a mono microphone
RATE = 48000  # Or adjust to match your microphone's sampling rate


# --- Feature Extraction Function ---
def extract_features(y, sr):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    return {
        "chroma_stft_mean": np.mean(chroma_stft),
        "chroma_stft_std": np.std(chroma_stft),
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_centroid_std": np.std(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "spectral_bandwidth_std": np.std(spectral_bandwidth),
        "rolloff_mean": np.mean(rolloff),
        "rolloff_std": np.std(rolloff),
        "zero_crossing_rate_mean": np.mean(zero_crossing_rate),
        "zero_crossing_rate_std": np.std(zero_crossing_rate),
        **{f"mfcc{i}_mean": np.mean(mfcc[i]) for i in range(mfcc.shape[0])},
        **{f"mfcc{i}_std": np.std(mfcc[i]) for i in range(mfcc.shape[0])},
    }


# --- Load Pre-trained Model ---
try:
    rf_model = load("finalized_model.sav.joblib")
    st.success("Pre-trained emotion detection model loaded successfully!")
except FileNotFoundError:
    st.error(
        "Error: 'finalized_model.sav.joblib' not found. Please ensure the model is in the same directory."
    )
    st.stop()  # Halt execution if model not found

# --- Circular Buffer (Simplified for Streamlit Integration) ---
buffer = []
BUFFER_LENGTH = 10  # Adjust this to control analysis length


# --- Main Function (Called by Streamlit) ---
def predict_emotion():
    global buffer

    try:
        # Initialize PyAudio stream (created within the function for continuous capture)
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        while True:
            data = stream.read(CHUNK_SIZE)
            buffer.append(data)

            if len(buffer) == BUFFER_LENGTH:
                # Extract audio data from the buffer
                frames = b"".join(buffer)
                y = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

                # Extract features
                features = extract_features(y, RATE)
                df = pd.DataFrame(features, index=[0])

                # Make prediction
                prediction = rf_model.predict(df)[0]

                # Display prediction on Streamlit interface
                st.write(f"Predicted Emotion: {prediction}")
                buffer = []  # Clear the buffer

    except KeyboardInterrupt:
        print("Exiting...")
        exit()
    finally:
        # Close audio stream resources
        stream.stop_stream()
        stream.close()
        p.terminate()


# --- Streamlit App ---
st.title("Real-Time Emotion Detection")
st.write("Speak into your microphone and see the predicted emotion!")

# Start emotion prediction on button click
if st.button("Start Prediction"):
    predict_emotion
