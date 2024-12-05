import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tf_keras import Input, Model
from tf_keras.src.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, Callback
from tf_keras.src.layers import Conv2D, MaxPooling2D, Flatten, Embedding, Concatenate, Dense, Dropout
from tf_keras.src.optimizers import Adam
from keras.src.saving import load_model

import os
import numpy as np
import pandas as pd
from tf_keras.src.regularizers import l2
from datetime import datetime
import shutil
import threading
from utils.utils import *
from utils.InterruptTraining import InterruptTraining

#PARAMS
label_dir = '../../recordings_with_tones.csv'
audio_dir = '../../recordings/stageI'
data_dir = "data"
load_data_from_file = True  # Ustaw na False, aby przetworzyć dane dźwięke na nowo, na True, aby je załadować z pliku
load_dir = "data/20241205_213929"
model_dir = "models"

input_shape = (63, 13, 1) #wcześniej było (13, 200, 1) 
word_index_to_name = {
    0: 'a0', 1: 'a1', 2: 'a2', 3: 'a3', 4: 'a4',
    5: 'a5', 6: 'a6', 7: 'a7', 8: 'a8', 9: 'a9',
    10: 'a10', 11: 'a100'
}

learning_rate=1e-3
batch_size=16

patience=20
min_delta=0.0001



# def build_pronunciation_model_with_word_info(num_words=12, embedding_dim=1):
#     # Audio input (MFCCs)
#     input_audio = Input(shape=(13, 200, 1))  # Assuming input is MFCC features of shape (13, 200)

#     # Word identifier input (word index as integer)
#     input_word = Input(shape=(1,), dtype='int32')  # Shape (1,) for single word index

#     # Convolutional layers for audio feature extraction
#     x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(input_audio)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(0.4)(x)  # Dropout layer to reduce overfitting
#     x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01))(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(0.4)(x)  # Dropout layer
#     x = Flatten()(x)

#     # Embedding layer to map word index to a vector representation
#     embedding = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=1)(input_word)
#     embedding = Flatten()(embedding)

#     # Concatenate the audio features and the word embedding
#     x = Concatenate()([x, embedding])

#     # Fully connected layers
#     x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
#     x = Dropout(0.4)(x)  # Dropout layer

#     # Output layer: single neuron for pronunciation quality (0 or 1)
#     output = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification

#     # Build the model
#     model = Model(inputs=[input_audio, input_word], outputs=output)
#     model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

#     return model

# def build_pronunciation_model_with_word_info_no_dropout(num_words=12, embedding_dim=10):
#     # Audio input (MFCCs)
#     input_audio = Input(shape=(13, 200, 1))

#     # Word identifier input (word index as integer)
#     input_word = Input(shape=(1,), dtype='int32')  # Shape (1,) for single word index

#     # Convolutional layers for audio feature extraction
#     x = Conv2D(32, (3, 3), activation='relu')(input_audio)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Conv2D(64, (3, 3), activation='relu')(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Flatten()(x)

#     # Embedding layer to map word index to a vector representation
#     embedding = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=1)(input_word)
#     embedding = Flatten()(embedding)

#     # Concatenate the audio features and the word embedding
#     x = Concatenate()([x, embedding])

#     # Fully connected layer
#     x = Dense(64, activation='relu')(x)

#     # Output layer: single neuron for pronunciation quality (0 or 1)
#     output = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification (well-pronounced or mispronounced)

#     # Build the model
#     model = Model(inputs=[input_audio, input_word], outputs=output)
#     model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

#     return model

def get_model(num_words=12, embedding_dim=10):
    input_audio = Input(shape=input_shape)
    input_word = Input(shape=(1,), dtype='int32') 
    x = Conv2D(64, (3,3), activation='relu', padding='same')(input_audio)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    #x = Dropout(0.4)(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    #x = Dropout(0.4)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    #x = Dropout(0.4)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    #x = Dropout(0.4)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    #x = Dropout(0.4)(x)
    
    x = Flatten()(x)
    embedding = Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=1)(input_word)
    embedding = Flatten()(embedding)
    x = Concatenate()([x, embedding])
    x = Dense(256, activation='relu')(x) 
    x = Dense(128, activation='relu')(x) 
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_audio, input_word], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model


X_train, X_val, X_test, word_train, word_val, word_test, y_train, y_val, y_test = (None,)*9
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


if not load_data_from_file:
    labels = load_labels(label_dir)
    save_dir = os.path.join(data_dir, timestamp)
    
    subsets = ["train", "val", "test"]
    for subset in subsets:
        os.makedirs(os.path.join(save_dir, subset), exist_ok=True)
    
    X, word_indices, y = load_data_conv(audio_dir, labels, 200)
    X_train, X_test, word_train, word_test, y_train, y_test = train_test_split(X, word_indices, y, test_size=0.10, random_state=42)
    X_train, X_val, word_train, word_val, y_train, y_val = train_test_split(X_train, word_train, y_train, test_size=0.111, random_state=42)
    
    # Zapis danych
    save_data({"X_train": X_train, "y_train": y_train, "word_train": word_train}, save_dir, "train")
    save_data({"X_val": X_val, "y_val": y_val, "word_val": word_val}, save_dir, "val")
    save_data({"X_test": X_test, "y_test": y_test, "word_test": word_test}, save_dir, "test")
    
    print(f"Dane zapisane w katalogu: {save_dir}")
else:
    # Ładowanie danych
    train_data = load_data(load_dir, "train")
    val_data = load_data(load_dir, "val")
    test_data = load_data(load_dir, "test")
    
    X_train, y_train, word_train = train_data["X_train"], train_data["y_train"], train_data["word_train"]
    X_val, y_val, word_val = val_data["X_val"], val_data["y_val"], val_data["word_val"]
    X_test, y_test, word_test = test_data["X_test"], test_data["y_test"], test_data["word_test"]

    print(f"Dane załadowane z katalogu: {load_dir}")
    
# Wyświetlanie wielkości zbiorów
print(f"Liczba próbek w zbiorze treningowym: {len(X_train)}")
print(f"Liczba próbek w zbiorze walidacyjnym: {len(X_val)}")
print(f"Liczba próbek w zbiorze testowym: {len(X_test)}")

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

model_save_dir =  os.path.join(model_dir, timestamp)
os.makedirs(model_save_dir, exist_ok=True)
train_logs_path = os.path.join(model_save_dir, 'train_logs')
os.makedirs(train_logs_path, exist_ok=True)

model_cur_dir =  os.path.join(model_dir, 'current')
if os.path.exists(model_cur_dir):
    shutil.rmtree(model_cur_dir)  
os.makedirs(model_cur_dir, exist_ok=True)
cur_train_logs_path = os.path.join(model_cur_dir, 'train_logs')
os.makedirs(cur_train_logs_path, exist_ok=True)
# Build the model
#model = build_pronunciation_model_with_word_info()
model = get_model()
save_model_info(model, os.path.join(model_save_dir, 'model_info.txt'))
best_checkpoint_dir= os.path.join(model_save_dir, f'best_checkpoint_model.h5')

#Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta, mode='min', restore_best_weights=True) #prevent overfitting
checkpoint = ModelCheckpoint(best_checkpoint_dir, save_best_only=True, monitor='val_loss', mode='min')
tensorboard_callback = TensorBoard(log_dir=train_logs_path, histogram_freq=1)
tensorboard_callback_cur = TensorBoard(log_dir=cur_train_logs_path, histogram_freq=1)
interrupt_callback = InterruptTraining()

# Train the model
history = model.fit(
    [X_train, word_train],
    y_train,
    epochs=100,  # Increased number of epochs to allow early stopping to work
    batch_size=batch_size,
    validation_data=([X_val, word_val], y_val),
    callbacks=[tensorboard_callback, tensorboard_callback_cur, checkpoint, interrupt_callback]
)

#model.save(os.path.join(model_save_dir, f'model.h5'))
load_model(best_checkpoint_dir)
# Make predictions on the test set
predictions = model.predict([X_test, word_test])
predicted_labels = (predictions > 0.5).astype(int)  # Threshold at 0.5



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
with open(os.path.join(model_save_dir, f'result.txt'), "w") as file:
    for word_index, accuracy in word_accuracies.items():
        word_name = word_index_to_name[word_index]
        result = f"Word {word_name}: Accuracy = {accuracy:.2f}"
        print(result)
        file.write(result + "\n")
