import os
import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tf_keras import Input, Model
from tf_keras.src.callbacks import EarlyStopping
from tf_keras.src.layers import Conv2D, MaxPooling2D, Flatten, Embedding, Concatenate, Dense, Dropout
from tf_keras.src.optimizers import Adam


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

                            word_column = f"{word_part}r"  # Column name (e.g., 'a0r')

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

# 3. Building the Model

def build_pronunciation_model_with_word_info(num_words=12, embedding_dim=1):
    # Audio input (MFCCs)
    input_audio = Input(shape=(13, 200, 1))  # Assuming input is MFCC features of shape (13, 200)

    # Word identifier input (word index as integer)
    input_word = Input(shape=(1,), dtype='int32')  # Shape (1,) for single word index

    # Convolutional layers for audio feature extraction
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(input_audio)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)  # Dropout layer to reduce overfitting
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)  # Dropout layer
    x = Flatten()(x)

    # Embedding layer to map word index to a vector representation
    embedding = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=1)(input_word)
    embedding = Flatten()(embedding)

    # Concatenate the audio features and the word embedding
    x = Concatenate()([x, embedding])

    # Fully connected layers
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)  # Dropout layer

    # Output layer: single neuron for pronunciation quality (0 or 1)
    output = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification

    # Build the model
    model = Model(inputs=[input_audio, input_word], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def build_pronunciation_model_with_word_info_no_dropout(num_words=12, embedding_dim=10):
    # Audio input (MFCCs)
    input_audio = Input(shape=(13, 200, 1))  # Assuming input is MFCC features of shape (13, 200)

    # Word identifier input (word index as integer)
    input_word = Input(shape=(1,), dtype='int32')  # Shape (1,) for single word index

    # Convolutional layers for audio feature extraction
    x = Conv2D(32, (3, 3), activation='relu')(input_audio)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    # Embedding layer to map word index to a vector representation
    embedding = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=1)(input_word)
    embedding = Flatten()(embedding)

    # Concatenate the audio features and the word embedding
    x = Concatenate()([x, embedding])

    # Fully connected layer
    x = Dense(64, activation='relu')(x)

    # Output layer: single neuron for pronunciation quality (0 or 1)
    output = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification (well-pronounced or mispronounced)

    # Build the model
    model = Model(inputs=[input_audio, input_word], outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Load the labels from CSV
labels = load_labels('Speech-Recognition-Model-for-Mandarin-Chinese/recordings_with_tones.csv')

# Load the data
X, word_indices, y = load_data_conv('recordings/stageI', labels, 200)

# Split the data into training and testing sets
X_train, X_test, word_train, word_test, y_train, y_test = train_test_split(X, word_indices, y, test_size=0.2, random_state=42)

# Build the model
#model = build_pronunciation_model_with_word_info()
model = build_pronunciation_model_with_word_info_no_dropout()
# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    [X_train, word_train],
    y_train,
    epochs=20,  # Increased number of epochs to allow early stopping to work
    batch_size=32,
    validation_data=([X_test, word_test], y_test),
    callbacks=[early_stopping]
)

model.save(f'noDropoutEmbeddingModel.h5')
# Make predictions on the test set
predictions = model.predict([X_test, word_test])
predicted_labels = (predictions > 0.5).astype(int)  # Threshold at 0.5

# Map word indices back to their names
word_index_to_name = {
    0: 'a0', 1: 'a1', 2: 'a2', 3: 'a3', 4: 'a4',
    5: 'a5', 6: 'a6', 7: 'a7', 8: 'a8', 9: 'a9',
    10: 'a10', 11: 'a100'
}

word_accuracies = {}

# Get the unique word indices
unique_words = np.unique(word_test)

for word in unique_words:
    # Get indices for the specific word
    word_indices = (word_test == word).flatten()

    # Extract true labels and predictions for this word
    y_true_word = y_test[word_indices]
    y_pred_word = predicted_labels[word_indices]

    # Compute accuracy for this word
    word_accuracy = accuracy_score(y_true_word, y_pred_word)
    word_accuracies[word] = word_accuracy

# Print accuracy for each word using the name mapping
for word_index, accuracy in word_accuracies.items():
    word_name = word_index_to_name[word_index]
    print(f"Word {word_name}: Accuracy = {accuracy:.2f}")
