import os
import numpy as np
import pandas as pd
import librosa

import os
import librosa
import numpy as np
import pandas as pd
from tf_keras.src.regularizers import l2



def load_data_conv(data_dir, labels, max_length=300):
    X, word_indices, y = [], [], []

    # Preprocessing helpers
    def remove_silence(audio, sr, top_db=20):
        non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
        return np.concatenate([audio[start:end] for start, end in non_silent_intervals])

    def normalize_audio(audio):
        return librosa.util.normalize(audio)

    def reduce_noise(audio, sr, noise_factor=0.005):
        noise = np.random.normal(0, noise_factor, audio.shape)
        return audio - noise  # Subtract noise for simplicity

    for id_folder in os.listdir(data_dir):
        id_path = os.path.join(data_dir, id_folder)

        if os.path.isdir(id_path):  # Only process directories
            for audio_file in os.listdir(id_path):
                if audio_file.endswith('.wav'):
                    file_path = os.path.join(id_path, audio_file)

                    try:
                        # Extract the word part from the filename (e.g., 'a0' -> 0)
                        word_part = audio_file[:-4]  # Remove '.wav'
                        if word_part.startswith('a') and word_part[1:].isdigit():
                            word_index = int(word_part[1:])

                            # Map `a100` to index 11
                            if word_index == 100:
                                word_index = 11

                            word_column = f"{word_part}p"  # Column name (e.g., 'a0p')

                            # Find the corresponding row in the labels DataFrame
                            person_id = int(id_folder)
                            label_row = labels[labels['id'] == person_id]

                            if not label_row.empty and word_column in label_row.columns:
                                label = label_row[word_column].values[0]

                                # Skip if the label is NULL/NaN
                                if label == "NULL" or pd.isna(label):
                                    print(f"Skipping unrated recording {file_path} (NULL label for {word_column}).")
                                    continue

                                # Load the audio file
                                audio, sr = librosa.load(file_path, sr=None)

                                # Apply preprocessing
                                audio = remove_silence(audio, sr)  # Remove silence
                                audio = reduce_noise(audio, sr)    # Reduce noise
                                audio = normalize_audio(audio)     # Normalize amplitude

                                # Extract MFCCs
                                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                                mfccs = librosa.util.fix_length(mfccs, size=max_length, axis=1)

                                X.append(mfccs)
                                word_indices.append(word_index)
                                y.append(label)
                            else:
                                print(f"Skipping file {file_path} (missing label for {word_column}).")
                        else:
                            print(f"Invalid file name format: {audio_file}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    # Convert to numpy arrays
    X = np.array(X)
    word_indices = np.array(word_indices)
    y = np.array(y)

    # Add channel dimension for Conv2D
    X = X[..., np.newaxis]  # Shape: (num_samples, n_mfcc, max_length, 1)
    print(f"Loaded {len(X)} samples with shape {X.shape}.")
    return X, word_indices, y
# 2. Loading Labels from CSV
def load_labels(csv_file):
    df = pd.read_csv(csv_file)
    # Extract only the id and the word column (0/1 ratings)
    df.columns = df.columns.str.strip()
    return df

def save_data(data, save_dir, subset):
    for key, value in data.items():
        np.save(os.path.join(save_dir, subset, f"{key}.npy"), value)

def load_data(load_dir, subset):
    files = ["X", "y", "word"]
    return {f"{file}_{subset}": np.load(os.path.join(load_dir, subset, f"{file}_{subset}.npy")) for file in files}

def save_model_info(model, save_dir):
    with open(save_dir, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))  
        f.write("\nModel Configuration:\n")
        f.write(str(model.get_config()))
        