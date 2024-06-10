@echo off
setlocal enabledelayedexpansion

rem Definiowanie folderu głównego
set "root_folder=.\recordings"

rem Przechodzenie przez wszystkie pliki .ogg w folderze głównym i podkatalogach
for /r "%root_folder%" %%f in (*.ogg) do (
    rem Pobranie ścieżki i nazwy pliku bez rozszerzenia
    set "input_file=%%f"
    set "output_file=%%~dpnf.mp3"
    
    rem Sprawdzenie, czy plik wyjściowy już istnieje
    if not exist "!output_file!" (
        rem Konwersja pliku .ogg na .wav
        ffmpeg -i "!input_file!" "!output_file!"
    ) else (
        echo Plik "!output_file!" już istnieje, pomijam konwersję.
    )
)

echo Konwersja zakończona.
pause
