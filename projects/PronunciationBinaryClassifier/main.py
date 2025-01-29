import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tf_keras import Input, Model
from tf_keras.src.backend import cast
from tf_keras.src.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, Callback
from tf_keras.src.layers import Conv2D, MaxPooling2D, Flatten, Embedding, Concatenate, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tf_keras.src.optimizers import Adam
from tf_keras.src.losses import binary_crossentropy, BinaryCrossentropy
from tf_keras.src.models import load_model
#from keras.src.saving import load_model
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tf_keras.src.regularizers import l2
from datetime import datetime
import shutil
import threading
from utils.utils import *
from utils.InterruptTraining import InterruptTraining

print("Available GPUs:")
print(tf.config.list_physical_devices('GPU'))

#PARAMS
load_dir = "data/a0_20250121_111131/spectrogram_augmented"
model_dir = "models"


#INPUT DATA
input_shape = (128, 72, 3)
word_index_to_name = {
    0: 'a0'
}
# word_index_to_name = {
#     0: 'a0', 1: 'a1', 2: 'a2', 3: 'a3', 4: 'a4',
#     5: 'a5', 6: 'a6', 7: 'a7', 8: 'a8', 9: 'a9',
#     10: 'a10', 11: 'a100'
# }

#HPARAMS
learning_rate=1e-4 #1e-3 1e-4
batch_size=32
dropout=0.2
smoothing=0.1
acceptance_threshold=0.5

patience=15
min_delta=0


def get_one_word_model(num_words=12, embedding_dim=10):
    input_audio = Input(shape=input_shape)
    input_word = Input(shape=(1,), dtype='int32') 
    x = Conv2D(64, (3,3), activation='relu', padding='same')(input_audio)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x) 
    x = Dropout(dropout)(x)
    x = Dense(128, activation='relu')(x) 
    x = Dropout(dropout)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_audio, input_word], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss=BinaryCrossentropy(label_smoothing=smoothing), 
        metrics=['accuracy']
    )
    return model


X_train, X_val, X_test, word_train, word_val, word_test, y_train, y_val, y_test = (None,)*9
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
model = get_one_word_model()
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
    callbacks=[tensorboard_callback, tensorboard_callback_cur, checkpoint, interrupt_callback, early_stopping]
)

#model.save(os.path.join(model_save_dir, f'model.h5'))
load_model(best_checkpoint_dir)
# Make predictions on the test set
predictions = model.predict([X_test, word_test])
predicted_labels = (predictions > acceptance_threshold).astype(int)



word_accuracies = {}
word_f1_scores = {}

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
    word_f1 = f1_score(y_true_word, y_pred_word, average='weighted')
    word_f1_scores[word] = word_f1

# Compute overall confusion matrix
overall_confusion_matrix = confusion_matrix(y_test.flatten(), predicted_labels.flatten())

# Reorder confusion matrix for Positive first
reordered_matrix = overall_confusion_matrix[[1, 0], :][:, [1, 0]]

# Save confusion matrix as an image
plt.figure(figsize=(8, 6))
sns.heatmap(reordered_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Positive", "Predicted Negative"], yticklabels=["Actual Positive", "Actual Negative"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
conf_matrix_path = os.path.join(model_save_dir, 'confusion_matrix.jpg')
plt.savefig(conf_matrix_path)
plt.close()

with open(os.path.join(model_save_dir, f'result.txt'), "w") as file:
    for word_index, accuracy in word_accuracies.items():
        word_name = word_index_to_name[word_index]
        f1_score_value = word_f1_scores[word_index]
        result = f"Word {word_name}: Accuracy = {accuracy:.2f}, F1 Score = {f1_score_value:.2f}"
        print(result)
        file.write(result + "\n")
        
with open(os.path.join(model_save_dir, f'hparamas.txt'), "w") as file:
    file.write(f"batch size: {batch_size}" + "\n")
    file.write(f"learning rate: {learning_rate}" + "\n")
    file.write(f"dropout: {dropout}" + "\n")
    file.write(f"smooting: {smoothing}" + "\n")
    file.write(f"acceptance threshold: {acceptance_threshold}" + "\n")