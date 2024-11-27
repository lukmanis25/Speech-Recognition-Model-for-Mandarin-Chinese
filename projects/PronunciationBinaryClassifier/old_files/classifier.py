import os

import joblib
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tf_keras import Sequential
from tf_keras.src.callbacks import EarlyStopping
from tf_keras.src.layers import Dense, Conv2D, MaxPooling2D, Flatten

accuracies = []
accuracies_conv = []
data_dir = 'recordings\stageI'
# Instead of loading the labels as before, now load continuous ratings
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

            if os.path.exists(file_path):  # Check if the file exists
                # Get the label from the dataframe based on ID
                label_row = labels[labels['id'] == int(id_folder)]
                #print(f"Label row for {id_folder}: {label_row}")  # Debug statement

                if not label_row.empty:
                    label = label_row[word].values[0]  # Extract the 0 rating
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


def TrainModel(data_dir, word, name):
    labels = load_labels("recordings_with_tones.csv", word)
    X,Y = load_data(data_dir, labels, name, word)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train a regression model (e.g., RandomForestRegressor)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    # Use RandomForestClassifier for binary classification
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class (1)

    # Convert probabilities to binary labels
    y_pred = (y_pred_proba > 0.55).astype(int)  # Apply threshold

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    # Print accuracy
    joblib.dump(model, f'audio_classification_modelSklearn{word}.joblib')
    #model.save(f'audio_classification_modelSklearn{word}.h5')
    print("word: " + name)
    print(f"Accuracy: {accuracy * 100:.2f}%")

def TrainModel_keras(data_dir, word, name):
    labels = load_labels("recordings_with_tones.csv", word)
    X,Y = load_data(data_dir, labels, name, word)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train a regression model (e.g., RandomForestRegressor)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # Input layer
    model.add(Dense(32, activation='relu'))  # Hidden layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_split=0.2, callbacks=[early_stopping])

    # Step 5: Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")
    accuracies.append(accuracy)
    # Step 6: Make predictions on the test set
    Y_pred = (model.predict(X_test) > 0.65).astype("int32")

    class_report = classification_report(y_test, Y_pred)

    print("Classification Report:\n", class_report)

    # Step 8: Save the model (optional)
    model.save(f'audio_classification_model{word}.h5')

def TrainModel_keras_conv(data_dir, word, name):
    labels = load_labels("recordings_with_tones.csv", word)
    X,Y = load_data_conv(data_dir, labels, name, word)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train a regression model (e.g., RandomForestRegressor)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=(X_train.shape[1], X_train.shape[2], 1)))  # 1 channel for grayscale
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_split=0.2, callbacks=[early_stopping])

    # Step 5: Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")
    accuracies_conv.append(accuracy)
    # Step 6: Make predictions on the test set
    Y_pred = (model.predict(X_test) > 0.5).astype("int32")

    class_report = classification_report(y_test, Y_pred)

    print("Classification Report:\n", class_report)

    # Step 8: Save the model (optional)
    model.save(f'audio_classification_modelconv{word}.h5')

TrainModel_keras_conv(data_dir, "a0r", "a0.wav")
TrainModel_keras_conv(data_dir, "a1r", "a1.wav")
TrainModel_keras_conv(data_dir, "a2r", "a2.wav")
TrainModel_keras_conv(data_dir, "a3r", "a3.wav")
TrainModel_keras_conv(data_dir, "a4r", "a4.wav")
TrainModel_keras_conv(data_dir, "a5r", "a5.wav")
TrainModel_keras_conv(data_dir, "a6r", "a6.wav")
TrainModel_keras_conv(data_dir, "a7r", "a7.wav")
TrainModel_keras_conv(data_dir, "a8r", "a8.wav")
TrainModel_keras_conv(data_dir, "a8r", "a8.wav")
TrainModel_keras_conv(data_dir, "a9r", "a9.wav")
TrainModel_keras_conv(data_dir, "a10r", "a10.wav")
TrainModel_keras_conv(data_dir, "a100r", "a100.wav")
TrainModel_keras(data_dir, "a0r", "a0.wav")
TrainModel_keras(data_dir, "a1r", "a1.wav")
TrainModel_keras(data_dir, "a2r", "a2.wav")
TrainModel_keras(data_dir, "a3r", "a3.wav")
TrainModel_keras(data_dir, "a4r", "a4.wav")
TrainModel_keras(data_dir, "a5r", "a5.wav")
TrainModel_keras(data_dir, "a6r", "a6.wav")
TrainModel_keras(data_dir, "a7r", "a7.wav")
TrainModel_keras(data_dir, "a8r", "a8.wav")
TrainModel_keras(data_dir, "a8r", "a8.wav")
TrainModel_keras(data_dir, "a9r", "a9.wav")
TrainModel_keras(data_dir, "a10r", "a10.wav")
TrainModel_keras(data_dir, "a100r", "a100.wav")
TrainModel(data_dir, "a0r", "a0.wav")
TrainModel(data_dir, "a1r", "a1.wav")
TrainModel(data_dir, "a2r", "a2.wav")
TrainModel(data_dir, "a3r", "a3.wav")
TrainModel(data_dir, "a4r", "a4.wav")
TrainModel(data_dir, "a5r", "a5.wav")
TrainModel(data_dir, "a6r", "a6.wav")
TrainModel(data_dir, "a7r", "a7.wav")
TrainModel(data_dir, "a8r", "a8.wav")
TrainModel(data_dir, "a9r", "a9.wav")
TrainModel(data_dir, "a10r", "a10.wav")
TrainModel(data_dir, "a100r", "a100.wav")

with open("accuracies65.txt", "w") as f:
    for j in range(10):
        print(f"Keras: {j} : {accuracies[j]}", file=f)
        print(f"Keras conv: {j} : {accuracies_conv[j]}", file=f)
        print(f"Sklearn: {j} : {accuracies[j + 10]}", file=f)
    print(f"Keras: 100 : {accuracies[11]}", file=f)
    print(f"Keras conv: 100 : {accuracies_conv[11]}", file=f)
    print(f"Sklearn: 100 : {accuracies[23]}", file=f)
