from pydub import AudioSegment
import pandas as pd
import os

def ogg_to_wav(input_path, output_path):
    # Wczytaj plik OGG
    print(input_path)
    sound = AudioSegment.from_ogg(input_path)
    
    # Zapisz jako plik WAV
    sound.export(output_path, format="wav")

def convert_directory(input_directory, output_directory):
    # Sprawdź czy ścieżki istnieją, jeśli nie, stwórz je
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Przetwórz każdy plik w katalogu wejściowym
    for filename in os.listdir(input_directory):
        if filename.endswith(".ogg"):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, os.path.splitext(filename)[0] + ".wav")
            ogg_to_wav(input_path, output_path)

# Wczytywanie pliku CSV
csv_file = 'recordings_with_tones.csv'
data = pd.read_csv(csv_file)

for index, row in data.iterrows():
    file_id = row['id']
    folder_path = f'C:/Projekty_magister/MBwI/Speech-Recognition-Model-for-Mandarin-Chinese/recordings/stageI/{file_id}'
    if os.path.exists(folder_path):
        output_directory = f'./recordings/stageI/{file_id}/'
        convert_directory(folder_path, output_directory)