#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

// Funkcja do odczytu obrazu z pliku tekstowego i normalizacji
std::vector<double> readImage(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<double> image;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            for (char pixel : line) {
                if (pixel == '0' || pixel == '1') {
                    image.push_back((double)(pixel - '0'));  // Normalizacja: 0 lub 1 zamieniamy na 0.0 lub 1.0
                }
            }
        }
        file.close();
    }
    return image; // Zwraca wektor 100 elementów (10x10).
}

// Prosta sieć neuronowa z ReLU
class SimpleNN {
public:
    SimpleNN(int input_size, int hidden_size, int output_size) {
        input_hidden_weights.resize(input_size * hidden_size);
        hidden_output_weights.resize(hidden_size * output_size);
        hidden_layer.resize(hidden_size);
        output_layer.resize(output_size);

        // Inicjalizacja wag losowymi wartościami
        for (double& weight : input_hidden_weights) {
            weight = ((double) rand() / RAND_MAX) * 2 - 1; // Wagi losowe w zakresie [-1, 1]
        }
        for (double& weight : hidden_output_weights) {
            weight = ((double) rand() / RAND_MAX) * 2 - 1; // Wagi losowe w zakresie [-1, 1]
        }
    }

    // Funkcja forward propagation z ReLU
    std::vector<double> forward(const std::vector<double>& inputs) {
        for (int i = 0; i < hidden_layer.size(); ++i) {
            hidden_layer[i] = 0;
            for (int j = 0; j < inputs.size(); ++j) {
                hidden_layer[i] += inputs[j] * input_hidden_weights[i * inputs.size() + j];
            }
            hidden_layer[i] = relu(hidden_layer[i]);  // Funkcja aktywacji ReLU
        }

        for (int i = 0; i < output_layer.size(); ++i) {
            output_layer[i] = 0;
            for (int j = 0; j < hidden_layer.size(); ++j) {
                output_layer[i] += hidden_layer[j] * hidden_output_weights[i * hidden_layer.size() + j];
            }
            output_layer[i] = sigmoid(output_layer[i]);  // Funkcja aktywacji sigmoid na wyjściu
        }

        return output_layer;
    }

    // Funkcja uczenia sieci (backpropagation)
    void train(const std::vector<double>& inputs, const std::vector<double>& expected_output, double learning_rate) {
        forward(inputs);

        std::vector<double> output_error(output_layer.size());
        for (int i = 0; i < output_layer.size(); ++i) {
            output_error[i] = expected_output[i] - output_layer[i];
        }

        std::vector<double> hidden_error(hidden_layer.size());
        for (int i = 0; i < hidden_layer.size(); ++i) {
            hidden_error[i] = 0;
            for (int j = 0; j < output_layer.size(); ++j) {
                hidden_error[i] += output_error[j] * hidden_output_weights[j * hidden_layer.size() + i];
            }
            hidden_error[i] *= relu_derivative(hidden_layer[i]);
        }

        // Aktualizacja wag między warstwą ukrytą a wyjściową
        for (int i = 0; i < hidden_output_weights.size(); ++i) {
            hidden_output_weights[i] += learning_rate * output_error[i % output_layer.size()] * hidden_layer[i / output_layer.size()];
        }

        // Aktualizacja wag między wejściem a warstwą ukrytą
        for (int i = 0; i < input_hidden_weights.size(); ++i) {
            input_hidden_weights[i] += learning_rate * hidden_error[i % hidden_layer.size()] * inputs[i / hidden_layer.size()];
        }
    }

private:
    std::vector<double> input_hidden_weights;
    std::vector<double> hidden_output_weights;
    std::vector<double> hidden_layer;
    std::vector<double> output_layer;

    // Funkcja aktywacji sigmoid
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Pochodna funkcji sigmoid
    double sigmoid_derivative(double x) {
        return x * (1.0 - x);
    }

    // Funkcja aktywacji ReLU
    double relu(double x) {
        return x > 0 ? x : 0;
    }

    // Pochodna funkcji ReLU
    double relu_derivative(double x) {
        return x > 0 ? 1 : 0;
    }
};

int main() {
    // Inicjalizacja sieci neuronowej
    SimpleNN nn(100, 30, 1); // Większa liczba neuronów ukrytych: 30

    // Obrazy treningowe i oczekiwane wyniki
    std::vector<std::string> training_files = {"training1.txt", "training2.txt", "training3.txt"};
    std::vector<int> expected_outputs = {1, 0, 0}; // obraz1 (X) -> 1, obraz2 (prostokąt) -> 0, obraz3 (O) -> 0

    // Trenowanie na obrazach
    for (int epoch = 0; epoch < 20000; ++epoch) {  // Zwiększona liczba epok do 20 000
        for (int i = 0; i < training_files.size(); ++i) {
            std::vector<double> image = readImage(training_files[i]);
            std::vector<double> expected_output = {static_cast<double>(expected_outputs[i])}; // Jedno wyjście per obraz
            nn.train(image, expected_output, 0.01); // Trenuj na każdym obrazie
        }
    }

    // Testowanie na obrazie "obraz.txt"
    std::vector<double> test_image = readImage("detection.txt");
    std::vector<double> result = nn.forward(test_image);

    // Wyświetlenie wyniku
    std::cout << "Wynik: " << result[0] << std::endl; // Powinno być blisko 1, jeśli obraz przedstawia kształt "X"

    return 0;
}

