import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from datetime import datetime

# Wczytywanie danych
X = np.load('./data/train/X_train.npy')
y = np.load('./data/train/y_train.npy')

# Podział danych na zbiór treningowy i walidacyjny
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Definicja modelu
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Inicjalizacja wykresów
plt.figure(figsize=(12, 6))

# Funkcja callback do zapisu modelu i rysowania wykresów po każdej epoce
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, plot_dir):
        super(CustomCallback, self).__init__()
        self.save_dir = save_dir
        self.plot_dir = plot_dir
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def on_epoch_end(self, epoch, logs=None):
        # Zapis modelu
        model_filename = f'model_{self.date_str}_epoch_{epoch + 1}.h5'
        model_path = os.path.join(self.save_dir, model_filename)
        self.model.save(model_path)
        print(f'Model saved at {model_path}')

        # Aktualizacja list strat i dokładności
        self.train_losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.train_accuracies.append(logs['accuracy'])
        self.val_accuracies.append(logs['val_accuracy'])

        # Rysowanie i zapisywanie wykresów
        plt.clf()

        # Wykres strat
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='train_loss')
        plt.plot(self.val_losses, label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Training and Validation Loss - Epoch {epoch + 1}')

        # Wykres dokładności
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='train_accuracy')
        plt.plot(self.val_accuracies, label='val_accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'Training and Validation Accuracy - Epoch {epoch + 1}')

        # Zapisywanie wykresów
        plt.tight_layout()
        plot_filename = f'epoch_plots.png'
        plot_path = os.path.join(self.plot_dir, plot_filename)
        plt.savefig(plot_path)
        print(f'Plots saved at {plot_path}')

# Tworzenie folderu na zapisane modele i wykresy, jeśli nie istnieje
model_dir = './saved_models'
os.makedirs(model_dir, exist_ok=True)

taining_plots_dir = './taining_plots'
os.makedirs(taining_plots_dir, exist_ok=True)

# Trenowanie modelu
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[CustomCallback(model_dir, taining_plots_dir)]
)

# Ostateczny wykres strat (loss) i dokładności (accuracy)
plt.clf()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
loss_plot_path = os.path.join(taining_plots_dir, 'final_loss_plot.png')
plt.savefig(loss_plot_path)

plt.clf()
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Over Epochs')
accuracy_plot_path = os.path.join(taining_plots_dir, 'final_accuracy_plot.png')
plt.savefig(accuracy_plot_path)

print(f'Final loss plot saved at {loss_plot_path}')
print(f'Final accuracy plot saved at {accuracy_plot_path}')
