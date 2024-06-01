# Przygotowanie projektu

Nagrania:
- nagrania pobieramu z dysku i wyodrębniamy w floderze głownym do catalogu './recordings'

Zamiana plików ogg na wav (windows):
- zainstalowac ffmpeg (https://phoenixnap.com/kb/ffmpeg-windows)
- uruchomić skrypt convert_ogg_to_wav.bat (może chwile potrwać)

# Projekty
Projekty znajdują się w 'projects' i tam tworzymu różne modele które testujemy lub implementujemy sami.
Najlepiej aby było ich jak najwięcej (nawet jak nie działają) żeby było co pokazać

Najlepiej aby projekt posiadał:
- Przygotowane dane do treningu w folderze 'projects/{nazwa_projektu}/data' (np cechy obrazu w pliku npy) tak aby szybko można było je wgrać na komputer zdalny. Dane podzielone na train i test.
- Plik 'projects/{nazwa_projektu}/train.py' który będzie zapisywał modele (najlepiej każdą epoke) oraz wykresy podczas uczenia aby było widać jak się uczy
- Plik 'projects/{nazwa_projektu}/test.py' który będzie testował przez zbiór testowy wybrany model (najlepszy z zapisanych) i zapisze wynikowe metryki do jakiegoś pliku
- Przykładowy projekt jest w ExampleNumberPronunciation