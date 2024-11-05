import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


csv_file = '../recordings_with_tones.csv'
audio_folder = 'C:/Projekty_magister_sem2/projekt_badawczy/Speech-Recognition-Model-for-Mandarin-Chinese/recordings'
output_folder = './recording_analysis_result'


keys = [
    'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a100',
    'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12',
    'q13', 'q14', 'q15', 'q16', 'q17', 'q18'
]


os.makedirs(output_folder, exist_ok=True)
data = pd.read_csv(csv_file)
word_result = []
tone_result = []

for index, row in data.iterrows():
    user_id = row['id']
    for key in keys:
        if pd.notnull(row[key+'p']):
            file_path_nums = f'{audio_folder}/stageI/{user_id}/{key[:2]}.wav'
            file_path_words = f'{audio_folder}/stageII/{user_id}/{key[:2]}.wav'
            if os.path.exists(file_path_nums) or os.path.exists(file_path_words):
                word_result.append({
                    'user_id': user_id,
                    'word_id': key,
                    'pronunciation_label': row[key+'p']
                })
                tone_key = key + 't'
                for syllable in [tone_key, tone_key+'1', tone_key+'2',tone_key+'3']:
                    if pd.notnull(row.get(syllable)):
                        tone_result.append({
                            'user_id': user_id,
                            'syllable_id': syllable,
                            'tone': row[syllable]
                        })


output_path = os.path.join(output_folder, 'word_id_counts.png')
word_counts = Counter(item['word_id'] for item in word_result)
plt.figure(figsize=(10, 6))
plt.bar(word_counts.keys(), word_counts.values(), width=0.6)
plt.xlabel('Word ID')
plt.ylabel('Liczba nagrań')
plt.title('Liczba nagrań dla każdego word_id')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout() 
plt.savefig(output_path)
plt.close()
print(f"Wykres zapisano w {output_path}")

df = pd.DataFrame(word_result)
grouped_data = df.groupby(['word_id', 'pronunciation_label']).size().unstack(fill_value=0)
output_path = os.path.join(output_folder, 'word_id_pronunciation_counts.png')
grouped_data.plot(kind='bar', figsize=(10, 6), color=['darkred', 'green'])
plt.xlabel('Word ID')
plt.ylabel('Liczba wystąpień')
plt.title('Liczba wystąpień dla każdego word_id według pronunciation_label')
plt.legend(title='Pronunciation Label', labels=['0', '1'])
plt.tight_layout()
plt.savefig(output_path)
plt.close()
print(f"Wykres zapisano w {output_path}")

output_path = os.path.join(output_folder, 'tone_counts.png')
tone_counts = Counter(item['tone'] for item in tone_result)
plt.figure(figsize=(10, 6))
plt.bar(tone_counts.keys(), tone_counts.values(), width=0.6)
plt.xlabel('Tone ID')
plt.ylabel('Liczba sylab')
plt.title('Liczba nagranych sylab dla danego tonu')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout() 
plt.savefig(output_path)
plt.close()
print(f"Wykres zapisano w {output_path}")