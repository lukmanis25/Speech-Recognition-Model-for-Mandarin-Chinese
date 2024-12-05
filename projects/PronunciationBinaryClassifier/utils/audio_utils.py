import pandas as pd
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

def display_mel_spectrogram(df: pd.DataFrame, additional_spectrogram_cols=[]):
    for i, row in df.iterrows():
        audio, sr = row['audio']
        S_dB = row['mel_spectrogram']
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        plt.show()
        
def display_linear_spectrogram(df: pd.DataFrame, hop_length=512, additional_spectrogram_cols=[]):
    for i, row in df.iterrows():
        audio, sr = row['audio']
        S = row['linear_spectrogram']
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
        plt.colorbar(label='Amplitude')
        plt.title("Linear Spectrogram")
        plt.tight_layout()
        plt.show()
        

def display_listenable_audio(df: pd.DataFrame, additional_audio_cols=[]):
    for i, row in df.iterrows():
        audio, sr = row['audio']
        print(f"Playing {row['path']} with label {row['label']}")
        ipd.display(ipd.Audio(data=audio, rate=sr))
        for col in additional_audio_cols:
            audiox, srx = row[col]
            print(col)
            ipd.display(ipd.Audio(data=audiox, rate=srx))

    