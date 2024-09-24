#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

// Funkcja do odczytu obrazu z pliku tekstowego i normalizacji
// Odczytuje obraz z pliku, gdzie wartości 0 lub 1 są przetwarzane
// i zapisywane w wektorze, reprezentującym piksele obrazu.
std::vector<double> readImage(const std::string& filename) {
    std::ifstream file(filename);  // Otwiera plik
    std::vector<double> image;     // Wektor do przechowywania wartości pikseli
    if (file.is_open()) {
        std::string line;
        // Odczyt kolejnych linii pliku
        while (std::getline(file, line)) {
            // Przetwarza każdy znak w linii
            for (char pixel : line) {
                // Sprawdza, czy piksel to 0 lub 1 i konwertuje go do typu double
                if (pixel == '0' || pixel == '1') {
                    image.push_back((double)(pixel - '0'));  // Zamienia '0' na 0.0 i '1' na 1.0
                }
            }
        }
        file.close();  // Zamknięcie pliku
    }
    return image;  // Zwraca wektor pikseli, reprezentujący obraz
}

// Prosta sieć neuronowa z warstwą ukrytą i aktywacją ReLU
class SimpleNN {
public:
    // Konstruktor sieci inicjalizuje wagi i rozmiary warstw
    SimpleNN(int input_size, int hidden_size, int output_size) {
        // Inicjalizacja wektorów wag między warstwami
        input_hidden_weights.resize(input_size * hidden_size); // Wagi między wejściem a warstwą ukrytą
        hidden_output_weights.resize(hidden_size * output_size); // Wagi między warstwą ukrytą a wyjściem
        hidden_layer.resize(hidden_size);  // Wektor reprezentujący warstwę ukrytą
        output_layer.resize(output_size);  // Wektor reprezentujący wyjście

        // Inicjalizacja wag losowymi wartościami w zakresie [-1, 1]
        for (double& weight : input_hidden_weights) {
            weight = ((double) rand() / RAND_MAX) * 2 - 1;
        }
        for (double& weight : hidden_output_weights) {
            weight = ((double) rand() / RAND_MAX) * 2 - 1;
        }
    }

    // Funkcja do propagacji w przód (forward propagation) z funkcją aktywacji ReLU
    std::vector<double> forward(const std::vector<double>& inputs) {
        // Obliczanie wartości neuronów w warstwie ukrytej
        for (int i = 0; i < hidden_layer.size(); ++i) {
            hidden_layer[i] = 0;
            for (int j = 0; j < inputs.size(); ++j) {
                // Oblicza sumę ważoną wejść i wag dla każdego neuronu ukrytego
                hidden_layer[i] += inputs[j] * input_hidden_weights[i * inputs.size() + j];
            }
            hidden_layer[i] = relu(hidden_layer[i]);  // Zastosowanie funkcji aktywacji ReLU
        }

        // Obliczanie wartości neuronów na wyjściu
        for (int i = 0; i < output_layer.size(); ++i) {
            output_layer[i] = 0;
            for (int j = 0; j < hidden_layer.size(); ++j) {
                // Oblicza sumę ważoną neuronów ukrytych i wag dla neuronów wyjściowych
                output_layer[i] += hidden_layer[j] * hidden_output_weights[i * hidden_layer.size() + j];
            }
            output_layer[i] = sigmoid(output_layer[i]);  // Zastosowanie funkcji aktywacji sigmoid
        }

        return output_layer;  // Zwraca wartości wyjściowe
    }

    // Funkcja do uczenia sieci metodą backpropagation
    void train(const std::vector<double>& inputs, const std::vector<double>& expected_output, double learning_rate) {
        // Propagacja w przód
        forward(inputs);

        // Obliczenie błędu na wyjściu (różnica między oczekiwanym a rzeczywistym wynikiem)
        std::vector<double> output_error(output_layer.size());
        for (int i = 0; i < output_layer.size(); ++i) {
            output_error[i] = expected_output[i] - output_layer[i];
        }

        // Obliczenie błędu dla warstwy ukrytej (propagacja błędu wstecz)
        std::vector<double> hidden_error(hidden_layer.size());
        for (int i = 0; i < hidden_layer.size(); ++i) {
            hidden_error[i] = 0;
            for (int j = 0; j < output_layer.size(); ++j) {
                hidden_error[i] += output_error[j] * hidden_output_weights[j * hidden_layer.size() + i];
            }
            hidden_error[i] *= relu_derivative(hidden_layer[i]);  // Zastosowanie pochodnej funkcji ReLU
        }

        // Aktualizacja wag między warstwą ukrytą a wyjściem na podstawie błędu
        for (int i = 0; i < hidden_output_weights.size(); ++i) {
            hidden_output_weights[i] += learning_rate * output_error[i % output_layer.size()] * hidden_layer[i / output_layer.size()];
        }

        // Aktualizacja wag między wejściem a warstwą ukrytą na podstawie błędu
        for (int i = 0; i < input_hidden_weights.size(); ++i) {
            input_hidden_weights[i] += learning_rate * hidden_error[i % hidden_layer.size()] * inputs[i / hidden_layer.size()];
        }
    }

private:
    std::vector<double> input_hidden_weights;  // Wagi między wejściem a warstwą ukrytą
    std::vector<double> hidden_output_weights; // Wagi między warstwą ukrytą a wyjściem
    std::vector<double> hidden_layer;          // Neurony warstwy ukrytej
    std::vector<double> output_layer;          // Neurony warstwy wyjściowej

    // Funkcja aktywacji sigmoid
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Funkcja aktywacji ReLU
    double relu(double x) {
        return x > 0 ? x : 0;
    }

    // Pochodna funkcji ReLU (używana w backpropagation)
    double relu_derivative(double x) {
        return x > 0 ? 1 : 0;
    }
};

int main() {
    // Inicjalizacja sieci neuronowej z 100 wejściami, 30 neuronami ukrytymi i 1 wyjściem
    SimpleNN nn(100, 30, 1);  // 100 pikseli obrazu wejściowego (10x10), 30 neuronów w warstwie ukrytej

    // Pliki z danymi treningowymi (obrazy) i oczekiwane wyniki
    std::vector<std::string> training_files = {"training1.txt", "training2.txt", "training3.txt"};
    std::vector<int> expected_outputs = {1, 0, 0};  // Oczekiwane wyniki: 1 dla "X", 0 dla innych kształtów

    // Trenowanie sieci neuronowej przez 20 000 epok
    for (int epoch = 0; epoch < 20000; ++epoch) {
        for (int i = 0; i < training_files.size(); ++i) {
            std::vector<double> image = readImage(training_files[i]);  // Wczytuje obraz z pliku
            std::vector<double> expected_output = {static_cast<double>(expected_outputs[i])};  // Oczekiwany wynik
            nn.train(image, expected_output, 0.01);  // Uczy sieć dla każdego obrazu
        }
    }

    // Testowanie sieci na nowym obrazie
    std::vector<double> test_image = readImage("detection.txt");  // Wczytuje obraz testowy
    std::vector<double> result = nn.forward(test_image);  // Przepuszcza obraz przez sieć

    // Wyświetlenie wyniku
    std::cout << "Wynik: " << result[0] << std::endl;  // Wyświetla wynik. Jeśli obraz to "X", wynik powinien być bliski 1.

    return 0;
}

