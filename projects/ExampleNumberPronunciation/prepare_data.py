import pandas as pd
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split

# Wczytywanie pliku CSV
csv_file = '../../recordings_with_tones.csv'
data = pd.read_csv(csv_file)

# Listy na cechy i etykiety
X = []
y = []

# Kolumny, które nas interesują
columns = ['a0p', 'a1p', 'a2p', 'a3p', 'a4p', 'a5p', 'a6p', 'a7p', 'a8p', 'a9p', 'a10p', 'a100p']

# Funkcja do ekstrakcji cech
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Iteracja po wierszach w pliku CSV
for index, row in data.iterrows():
    file_id = row['id']
    for column in columns:
        if pd.notnull(row[column]):
            file_path = f'../../recordings/stageI/{file_id}/{column[:2]}.wav'
            if os.path.exists(file_path):
                features = extract_features(file_path)
                X.append(features)
                y.append(row[column])

# Konwersja list do numpy arrays
X = np.array(X)
y = np.array(y)

# Podział danych na zbiór treningowy i walidacyjny
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie folderów, jeśli nie istnieją
os.makedirs('./data/train/', exist_ok=True)
os.makedirs('./data/test/', exist_ok=True)

# Zapisanie danych jako pliki numpy
np.save('./data/train/X_train.npy', X_train)
np.save('./data/train/y_train.npy', y_train)
np.save('./data/test/X_test.npy', X_test)
np.save('./data/test/y_test.npy', y_test)

print("Dane zostały przygotowane i zapisane.")
