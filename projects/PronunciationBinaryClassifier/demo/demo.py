import os
import numpy as np
import tensorflow as tf
import argparse
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import butter, lfilter
from noisereduce import reduce_noise
from tf_keras.src.models import load_model

# Ścieżka do najlepszego zapisanego modelu
MODEL_PATH = "../models/best_a0_20250121_143611/best_checkpoint_model.h5"
INPUT_SHAPE = (128, 72, 3)  # Musi pasować do kształtu wejściowego modelu
ACCEPTANCE_THRESHOLD = 0.5  # Próg akceptacji
TARGET_SR = 16000
NOISE_PROFILE = 0.5
LOWCUT = 80
HIGHCUT = 8000
TOP_DB = 30
MAX_LENGTH = 36523

# Wczytanie modelu
print("Ładowanie modelu...")
model = load_model(MODEL_PATH)
print("Model załadowany.")

def remove_silence(audio, sr, top_db=TOP_DB):
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

def reduce_noise_in_audio(audio, sr):
    noise_part = audio[:int(NOISE_PROFILE * sr)]
    return reduce_noise(y=audio, sr=sr, y_noise=noise_part)

def butter_bandpass_filter(audio, sr):
    nyquist = 0.5 * sr
    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist
    b, a = butter(1, [low, high], btype='band')
    return lfilter(b, a, audio)

def spectral_noise_reduction(audio, sr):
    stft = librosa.stft(audio)
    magnitude, phase = librosa.magphase(stft)
    noise_threshold = np.mean(magnitude) * 0.5
    reduced_magnitude = np.where(magnitude > noise_threshold, magnitude, 0)
    return librosa.istft(reduced_magnitude * phase)

def normalize_audio(audio):
    return librosa.util.normalize(audio)

def resample_audio(audio, sr):
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio, TARGET_SR

def pad_to_max_length(audio, max_length=MAX_LENGTH):
    if len(audio) < max_length:
        return np.pad(audio, (0, max_length - len(audio)), mode='constant')
    elif len(audio) > max_length:
        return audio[:max_length]
    return audio

def generate_mel_spectrogram(audio, sr, n_mels=128):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def prepare_spectrogram_images(mel_specs, cmap='viridis'):
    colormap = cm.get_cmap(cmap)
    mel_specs_normalized = (mel_specs - np.min(mel_specs)) / (np.max(mel_specs) - np.min(mel_specs) + 1e-6)
    images = (colormap(mel_specs_normalized)[..., :3] * 255).astype(np.uint8)
    return images

def load_audio(file_path):
    """
    Przetwarza plik audio na odpowiedni format dla modelu.
    """
    audio, sr = librosa.load(file_path, sr=None)
    audio = remove_silence(audio, sr)
    audio = reduce_noise_in_audio(audio, sr)
    audio = butter_bandpass_filter(audio, sr)
    audio = spectral_noise_reduction(audio, sr)
    audio = normalize_audio(audio)
    audio, sr = resample_audio(audio, sr)
    audio = pad_to_max_length(audio, MAX_LENGTH)
    
    # Zapisz przetworzony plik audio
    processed_audio_path = file_path.replace(".wav", "_processed.wav")
    sf.write(processed_audio_path, audio, sr)
    print(f"Przetworzony plik audio zapisany jako: {processed_audio_path}")
    
    spectrogram = generate_mel_spectrogram(audio, sr)
    spectrogram_image = prepare_spectrogram_images(spectrogram)
    spectrogram_image = np.expand_dims(spectrogram_image, axis=0)  # Dodanie batch dimension
    return spectrogram_image

def predict_audio(file_path):
    """
    Dokonuje predykcji na podanym pliku audio.
    """
    spectrogram = load_audio(file_path)
    word_index = np.array([[0]])
    
    prediction = model.predict([spectrogram, word_index])
    probability = prediction[0][0]
    
    result = "POPRAWNE" if probability >= ACCEPTANCE_THRESHOLD else "NIEPOPRAWNE"
    print(f"Plik: {file_path} -> Wynik: {result} (Prawdopodobieństwo: {probability:.2f})")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo użycia modelu do oceny nagrania.")
    parser.add_argument("file", type=str, help="Ścieżka do pliku audio")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print("Podany plik nie istnieje.")
        exit(1)
    
    predict_audio(args.file)
