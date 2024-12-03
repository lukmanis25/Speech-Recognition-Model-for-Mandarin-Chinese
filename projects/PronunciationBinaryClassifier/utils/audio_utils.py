import pandas as pd
import IPython.display as ipd

def display_listenable_audio(df: pd.DataFrame, additional_audio_cols=[]):
    for i, row in df.iterrows():
        audio, sr = row['audio']
        print(f"Playing {row['path']} with label {row['label']}")
        ipd.display(ipd.Audio(data=audio, rate=sr))
        for col in additional_audio_cols:
            audiox, srx = row[col]
            print(col)
            ipd.display(ipd.Audio(data=audiox, rate=srx))

    