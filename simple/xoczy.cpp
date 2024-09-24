#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>

// Funkcja do odczytu obrazu z pliku tekstowego i konwersji na wektor 0 i 1
std::vector<int> readImage(const std::string& filename) {
    std::ifstream file(filename);  // Otwiera plik
    std::vector<int> image;        // Wektor przechowujący piksele obrazu
    if (file.is_open()) {          // Sprawdza, czy plik został poprawnie otwarty
        std::string line;
        // Odczytuje każdą linię z pliku
        while (std::getline(file, line)) {
            for (char pixel : line) {
                // Dodaje do wektora wartość 0 lub 1 na podstawie wartości znaku
                if (pixel == '0' || pixel == '1') {
                    image.push_back(pixel - '0');  // Konwersja znaku na liczbę 0 lub 1
                }
            }
        }
    }
    file.close();  // Zamknięcie pliku
    return image;  // Zwraca wektor 100 elementów (obraz 10x10)
}

/////////////////////////////////////////

// Funkcja aktywacji sigmoid, która przekształca wartość na zakres 0-1
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Funkcja zwracająca pochodną funkcji sigmoid, używana w procesie uczenia (backpropagation)
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Prosta sieć neuronowa
class SimpleNN {
public:
    // Konstruktor inicjalizujący sieć neuronową
    SimpleNN(int input_size, int hidden_size, int output_size) {
        // Inicjalizacja wektorów wag między warstwą wejściową a ukrytą oraz ukrytą a wyjściową
        input_hidden_weights.resize(input_size * hidden_size);
        hidden_output_weights.resize(hidden_size * output_size);
        hidden_layer.resize(hidden_size);  // Warstwa ukryta
        output_layer.resize(output_size);  // Warstwa wyjściowa

        // Losowa inicjalizacja wag
        for (double& weight : input_hidden_weights) {
            weight = (double) rand() / RAND_MAX;
        }
        for (double& weight : hidden_output_weights) {
            weight = (double) rand() / RAND_MAX;
        }
    }

    // Funkcja forward propagation: przekazuje dane wejściowe przez sieć
    std::vector<double> forward(const std::vector<int>& inputs) {
        // Oblicza wartości neuronów w warstwie ukrytej
        for (int i = 0; i < hidden_layer.size(); ++i) {
            hidden_layer[i] = 0;  // Zeruje bieżącą wartość neuronu ukrytego
            for (int j = 0; j < inputs.size(); ++j) {
                hidden_layer[i] += inputs[j] * input_hidden_weights[i * inputs.size() + j];  // Suma ważona
            }
            hidden_layer[i] = sigmoid(hidden_layer[i]);  // Funkcja aktywacji sigmoid
        }

        // Oblicza wartości neuronów w warstwie wyjściowej
        for (int i = 0; i < output_layer.size(); ++i) {
            output_layer[i] = 0;  // Zeruje bieżącą wartość neuronu wyjściowego
            for (int j = 0; j < hidden_layer.size(); ++j) {
                output_layer[i] += hidden_layer[j] * hidden_output_weights[i * hidden_layer.size() + j];  // Suma ważona
            }
            output_layer[i] = sigmoid(output_layer[i]);  // Funkcja aktywacji sigmoid
        }

        return output_layer;  // Zwraca wartości neuronów w warstwie wyjściowej
    }

    // Funkcja ucząca sieć neuronową (backpropagation)
    void train(const std::vector<int>& inputs, const std::vector<int>& expected_output, double learning_rate) {
        // Forward propagation
        forward(inputs);

        // Oblicza błąd wyjściowy
        std::vector<double> output_error(output_layer.size());
        for (int i = 0; i < output_layer.size(); ++i) {
            output_error[i] = expected_output[i] - output_layer[i];  // Różnica między oczekiwanym a rzeczywistym wynikiem
        }

        // Oblicza błąd warstwy ukrytej
        std::vector<double> hidden_error(hidden_layer.size());
        for (int i = 0; i < hidden_layer.size(); ++i) {
            hidden_error[i] = 0;
            for (int j = 0; j < output_layer.size(); ++j) {
                hidden_error[i] += output_error[j] * hidden_output_weights[j * hidden_layer.size() + i];  // Suma błędów dla ukrytych neuronów
            }
            hidden_error[i] *= sigmoid_derivative(hidden_layer[i]);  // Mnoży przez pochodną funkcji aktywacji
        }

        // Aktualizuje wagi między warstwą ukrytą a wyjściową
        for (int i = 0; i < hidden_output_weights.size(); ++i) {
            hidden_output_weights[i] += learning_rate * output_error[i % output_layer.size()] * hidden_layer[i / output_layer.size()];
        }

        // Aktualizuje wagi między wejściem a warstwą ukrytą
        for (int i = 0; i < input_hidden_weights.size(); ++i) {
            input_hidden_weights[i] += learning_rate * hidden_error[i % hidden_layer.size()] * inputs[i / hidden_layer.size()];
        }
    }

private:
    // Wektory wag oraz wartości neuronów dla warstwy ukrytej i wyjściowej
    std::vector<double> input_hidden_weights;
    std::vector<double> hidden_output_weights;
    std::vector<double> hidden_layer;
    std::vector<double> output_layer;
};

/////////////////////////////////////////////////

// Główna funkcja programu
int main() {
    // Inicjalizacja sieci neuronowej: 100 neuronów wejściowych (obraz 10x10), 15 neuronów ukrytych, 1 wyjściowy
    SimpleNN nn(100, 15, 1);

    // Obrazy treningowe i oczekiwane wyniki (np. 1 oznacza rozpoznanie wzoru, 0 brak wzoru)
    std::vector<std::string> training_files = {"training1.txt", "training2.txt", "training3.txt"};
    std::vector<int> expected_outputs = {1, 0, 0};  // Oczekiwane wyjścia dla obrazów

    // Trenowanie sieci neuronowej na danych treningowych
    for (int epoch = 0; epoch < 10000; ++epoch) {
        for (int i = 0; i < training_files.size(); ++i) {
            std::vector<int> image = readImage(training_files[i]);  // Odczytuje obraz z pliku
            std::vector<int> expected_output = {expected_outputs[i]};  // Oczekiwane wyjście
            nn.train(image, expected_output, 0.01);  // Uczy sieć z użyciem algorytmu backpropagation
        }
    }

    // Testowanie sieci na nowym obrazie "detection.txt"
    std::vector<int> test_image = readImage("detection.txt");
    std::vector<double> result = nn.forward(test_image);  // Forward propagation dla testowego obrazu

    // Wyświetlenie wyniku (blisko 1 oznacza rozpoznanie wzoru, blisko 0 brak rozpoznania)
    std::cout << "Wynik: " << result[0] << std::endl;

    return 0;
}

