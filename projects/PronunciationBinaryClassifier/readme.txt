Uruchomienie:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt - do zainstalowania paczek jak nie ma
python main.py


Podgląd na żywo treningu (trzba timestamp zmienić):
tensorboard --logdir=models\current\train_logs 

q + enter aby zakończyć trening i wykonać test