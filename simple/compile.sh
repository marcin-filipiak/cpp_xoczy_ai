#!/bin/sh

# Usuwanie istniejącego pliku programu, jeśli istnieje
if [ -f xoczy ]; then
  rm xoczy
fi

# Usuwanie pliku error_log.txt, jeśli istnieje
if [ -f error_log.txt ]; then
  rm error_log.txt
fi

echo "----Kompilacja programu----"

# Kompilacja z przekierowaniem błędów do pliku error_log.txt
g++ -o xoczy xoczy.cpp 2> error_log.txt

# Sprawdzanie, czy plik error_log.txt jest pusty
if [ -s error_log.txt ]; then
  echo "---Błędy podczas kompilacji---"
  # Wyświetlanie błędów na ekranie z paginacją
  more error_log.txt
else
  echo "---Kompilacja ukończona pomyślnie---"
  echo "---Uruchamianie programu---"
  chmod +x xoczy
  ./xoczy
  echo "---Zakończono program---"
  
  # Usunięcie pliku error_log.txt, ponieważ nie było błędów
  rm error_log.txt
fi

