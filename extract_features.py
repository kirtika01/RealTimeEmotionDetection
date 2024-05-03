import librosa
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Path to the audio file')
parser.add_argument('audio_path', type=str, help='Path to the audio file')
args = parser.parse_args()

def extract_features(y, sr):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    return {
        'chroma_stft_mean': np.mean(chroma_stft),
        'chroma_stft_std': np.std(chroma_stft),
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_std': np.std(spectral_centroid),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'spectral_bandwidth_std': np.std(spectral_bandwidth),
        'rolloff_mean': np.mean(rolloff),
        'rolloff_std': np.std(rolloff),
        'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
        'zero_crossing_rate_std': np.std(zero_crossing_rate),
        **{f'mfcc{i}_mean': np.mean(mfcc[i]) for i in range(mfcc.shape[0])},
        **{f'mfcc{i}_std': np.std(mfcc[i]) for i in range(mfcc.shape[0])}
    }

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

if os.path.isdir(args.audio_path):
    wav_files = [file for file in os.listdir(args.audio_path) if file.endswith('.wav')]
    i = 1
    for wav_file in wav_files:
        audio_path = os.path.join(args.audio_path, wav_file)
        
        y, sr = librosa.load(audio_path)
        parts = audio_path.split('-')
        label = parts[2]

        features = extract_features(y, sr)
        
        df = pd.DataFrame(features, index=[0])
        df['label'] = emotion_map[label]
        
        # if i == 1:
        #     df.to_csv('audio_features.csv', index=False)
        #     i = 0
        #     continue
        
        df.to_csv('audio_features.csv', index=False, mode='a', header=False)
        
       
else:
    y, sr = librosa.load(args.audio_path)
    
    features = extract_features(y, sr)
    
    df = pd.DataFrame(features, index=[0])
    
    df.to_csv('audio_features.csv', index=False, mode='a')
    

