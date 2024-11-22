import os

from keras.src.saving import load_model
from sklearn.metrics import accuracy_score
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import logging
import csv

from tensorflow.python.ops.metrics_impl import accuracy
from tf_keras.src.callbacks import EarlyStopping

# Ignore all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")
level = 60
# Suppress absl warnings specifically
logging.getLogger("absl").setLevel(logging.ERROR)
data_dir = 'recordings\stageI'
accuracies = []
filename="accuracies.csv"
filename_conv="accuracies_conv.csv"
def load_labels(csv_file, word):
    df = pd.read_csv(csv_file)
    # Extract only the a0r ratings and IDs
    labels = df[['id', word]]
    return labels

def load_data(data_dir, labels, name, word):
    X, y = [], []

    for id_folder in os.listdir(data_dir):
        id_path = os.path.join(data_dir, id_folder)
        if os.path.isdir(id_path):  # Check if it's a directory
            file_path = os.path.join(id_path, name)  # Each folder contains 'a0.wav'
            #print(f"Checking folder: {id_folder}, looking for file: {file_path}")  # Debug statement

            if os.path.exists(file_path):  # Check if the file exists
                # Get the label from the dataframe based on ID
                label_row = labels[labels['id'] == int(id_folder)]
                #print(f"Label row for {id_folder}: {label_row}")  # Debug statement

                if not label_row.empty:
                    label = label_row[word].values[0]  # Extract the a0r rating
                    audio, sr = librosa.load(file_path, sr=None)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    X.append(np.mean(mfccs.T, axis=0))
                    y.append(label)

    print(f"Loaded {len(X)} samples.")  # Debug statement
    return np.array(X), np.array(y)

def load_data_conv(data_dir, labels, name, word, max_length=200):
    X, y = [], []

    for id_folder in os.listdir(data_dir):
        id_path = os.path.join(data_dir, id_folder)
        if os.path.isdir(id_path):  # Check if it's a directory
            file_path = os.path.join(id_path, name)  # Each folder contains 'a0.wav'

            if os.path.exists(file_path):  # Check if the file exists
                # Get the label from the dataframe based on ID
                label_row = labels[labels['id'] == int(id_folder)]

                if not label_row.empty:
                    label = label_row[word].values[0]  # Extract the label
                    audio, sr = librosa.load(file_path, sr=None)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

                    # Pad or truncate MFCC to ensure fixed length along the time axis
                    mfccs = librosa.util.fix_length(mfccs, size=max_length, axis=1)

                    X.append(mfccs)
                    y.append(label)

    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Add a channel dimension to X for compatibility with Conv2D
    X = X[..., np.newaxis]  # Shape: (num_samples, n_mfcc, max_length, 1)

    print(f"Loaded {len(X)} samples with shape {X.shape}.")  # Debug statement
    return X, y

def accuracy_meter(path,name, word):
    model = load_model(path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    labels = load_labels("recordings_with_tones.csv", word)
    X, Y = load_data(data_dir, labels, name, word)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    y_pred = (model.predict(X_test) > (level/100)).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    #print("Model Accuracy:", accuracy)
    print(f"{word} Model Accuracy:", accuracy)


def accuracy_meter_conv(path,name, word):
    labels = load_labels("recordings_with_tones.csv", word)
    X,Y = load_data_conv(data_dir, labels, name, word)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = load_model(path)
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    y_pred_proba= model.predict(X_test)
    #y_pred = np.argmax(y_pred_proba, axis=1)  # Adjust for multiclass

    y_pred = (model.predict(X_test) > (level / 100)).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    # print("Model Accuracy:", accuracy)
    print(f"{word} Model Accuracy:", accuracy)


accuracy_meter("audio_classification_modela0r.h5", "a0.wav", "a0r")
accuracy_meter("audio_classification_modela1r.h5", "a1.wav", "a1r")
accuracy_meter("audio_classification_modela2r.h5", "a2.wav", "a2r")
accuracy_meter("audio_classification_modela3r.h5", "a3.wav", "a3r")
accuracy_meter("audio_classification_modela4r.h5",  "a4.wav", "a4r")
accuracy_meter("audio_classification_modela5r.h5",  "a5.wav", "a5r")
accuracy_meter("audio_classification_modela6r.h5",  "a6.wav", "a6r")
accuracy_meter("audio_classification_modela7r.h5",  "a7.wav", "a7r")
accuracy_meter("audio_classification_modela8r.h5", "a8.wav", "a8r")
accuracy_meter("audio_classification_modela9r.h5",  "a9.wav", "a9r")
accuracy_meter("audio_classification_modela10r.h5",  "a10.wav", "a10r")
accuracy_meter("audio_classification_modela100r.h5",  "a100.wav", "a100r")


# accuracy_meter_conv("audio_classification_modelconva0r.h5", "a0.wav", "a0r")
# accuracy_meter_conv("audio_classification_modelconva1r.h5", "a1.wav", "a1r")
# accuracy_meter_conv("audio_classification_modelconva2r.h5", "a2.wav", "a2r")
# accuracy_meter_conv("audio_classification_modelconva3r.h5", "a3.wav", "a3r")
# accuracy_meter_conv("audio_classification_modelconva4r.h5",  "a4.wav", "a4r")
# accuracy_meter_conv("audio_classification_modelconva5r.h5",  "a5.wav", "a5r")
# accuracy_meter_conv("audio_classification_modelconva6r.h5",  "a6.wav", "a6r")
# accuracy_meter_conv("audio_classification_modelconva7r.h5",  "a7.wav", "a7r")
# accuracy_meter_conv("audio_classification_modelconva8r.h5", "a8.wav", "a8r")
# accuracy_meter_conv("audio_classification_modelconva9r.h5",  "a9.wav", "a9r")
# accuracy_meter_conv("audio_classification_modelconva10r.h5",  "a10.wav", "a10r")
# accuracy_meter_conv("audio_classification_modelconva100r.h5",  "a100.wav", "a100r")

# Function to append new data to the next available column
def update_csv_with_new_column(filename):
    try:
        # Read the existing CSV file to check the number of columns and rows
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)

        # Check how many columns are already in the file
        num_columns = len(rows[0])  # The number of columns in the first row (header)

        # Open the CSV file in write mode to update it
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Add the new level_val to the first row in the next available column
            rows[0].append(level)  # Add the new level value in the first row under a new column

            # Write the updated first row (with the new level value)
            writer.writerow(rows[0])

            # Now, append the new accuracy values under the new level column
            for i in range(len(accuracies)):
                # If there aren't enough rows in the CSV, add empty rows
                while len(rows) <= i + 1:
                    rows.append([''] * (num_columns + 1))  # Add empty rows to accommodate new column

                # Set the accuracy value under the new column
                rows[i + 1].append(accuracies[i])  # Append new values under the new level column

            # Write all the updated rows back to the file
            writer.writerows(rows[1:])


    except FileNotFoundError:
        # If the file doesn't exist, create it and write the first column (level_val and array_values)
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the level value in the first row, first column
            writer.writerow([level])

            # Write each value from array_values in a new row in the first column
            for value in accuracies:
                writer.writerow([value])


update_csv_with_new_column(filename)
#update_csv_with_new_column(filename_conv)