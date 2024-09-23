#!/bin/sh

# Usuwanie istniejącego pliku main, jeśli istnieje
if [ -f main_simple ]; then
  rm main_simple
fi

# Usuwanie pliku error_log.txt, jeśli istnieje
if [ -f error_log.txt ]; then
  rm error_log.txt
fi

echo "----Kompilacja programu----"

# Kompilacja z przekierowaniem błędów do pliku error_log.txt
g++ -o main_simple main_simple.cpp 2> error_log.txt

# Sprawdzanie, czy plik error_log.txt jest pusty
if [ -s error_log.txt ]; then
  echo "---Błędy podczas kompilacji---"
  # Wyświetlanie błędów na ekranie z paginacją
  more error_log.txt
else
  echo "---Kompilacja ukończona pomyślnie---"
  echo "---Uruchamianie programu---"
  chmod +x main_simple
  ./main_simple
  echo "---Zakończono program---"
  
  # Usunięcie pliku error_log.txt, ponieważ nie było błędów
  rm error_log.txt
fi

