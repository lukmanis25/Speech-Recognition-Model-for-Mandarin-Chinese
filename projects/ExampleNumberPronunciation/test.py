import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Wczytywanie danych testowych
X_test = np.load('./data/test/X_test.npy')
y_test = np.load('./data/test/y_test.npy')

# Wczytywanie modelu
model_path = './saved_models/model_2024-06-01_12-01-50_epoch_50.h5'  # Wstaw ścieżkę do wybranego modelu
model = tf.keras.models.load_model(model_path)

# Wykonywanie predykcji na danych testowych
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Obliczanie metryk
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

# Wyświetlanie wyników
print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1 Score:", f1)

# Zapisywanie wyników do pliku
results_dir = './test_results'
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, 'test_results.txt')

with open(results_file, 'w') as f:
    f.write(f'Test Accuracy: {accuracy}\n')
    f.write(f'Test Precision: {precision}\n')
    f.write(f'Test Recall: {recall}\n')
    f.write(f'Test F1 Score: {f1}\n')

print(f'Results saved to {results_file}')
