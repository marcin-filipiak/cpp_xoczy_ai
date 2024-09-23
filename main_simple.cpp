#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>


std::vector<int> readImage(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<int> image;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            for (char pixel : line) {
                if (pixel == '0' || pixel == '1') {
                    image.push_back(pixel - '0');
                }
            }
        }
    }
    file.close();
    return image; // Zwraca wektor 100 elementów (10x10).
}

/////////////////////////////////////////


// Funkcja aktywacji sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Pochodna funkcji sigmoid
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Prosta sieć neuronowa
class SimpleNN {
public:
    SimpleNN(int input_size, int hidden_size, int output_size) {
        // Inicjalizacja wag
        input_hidden_weights.resize(input_size * hidden_size);
        hidden_output_weights.resize(hidden_size * output_size);
        hidden_layer.resize(hidden_size);
        output_layer.resize(output_size);

        // Inicjalizacja wag losowymi wartościami
        for (double& weight : input_hidden_weights) {
            weight = (double) rand() / RAND_MAX;
        }
        for (double& weight : hidden_output_weights) {
            weight = (double) rand() / RAND_MAX;
        }
    }

    // Funkcja forward propagation
    std::vector<double> forward(const std::vector<int>& inputs) {
        // Obliczenia warstwy ukrytej
        for (int i = 0; i < hidden_layer.size(); ++i) {
            hidden_layer[i] = 0;
            for (int j = 0; j < inputs.size(); ++j) {
                hidden_layer[i] += inputs[j] * input_hidden_weights[i * inputs.size() + j];
            }
            hidden_layer[i] = sigmoid(hidden_layer[i]);
        }

        // Obliczenia warstwy wyjściowej
        for (int i = 0; i < output_layer.size(); ++i) {
            output_layer[i] = 0;
            for (int j = 0; j < hidden_layer.size(); ++j) {
                output_layer[i] += hidden_layer[j] * hidden_output_weights[i * hidden_layer.size() + j];
            }
            output_layer[i] = sigmoid(output_layer[i]);
        }

        return output_layer;
    }

    // Funkcja uczenia sieci (backpropagation)
    void train(const std::vector<int>& inputs, const std::vector<int>& expected_output, double learning_rate) {
        // Forward propagation
        forward(inputs);

        // Błąd wyjścia
        std::vector<double> output_error(output_layer.size());
        for (int i = 0; i < output_layer.size(); ++i) {
            output_error[i] = expected_output[i] - output_layer[i];
        }

        // Błąd warstwy ukrytej
        std::vector<double> hidden_error(hidden_layer.size());
        for (int i = 0; i < hidden_layer.size(); ++i) {
            hidden_error[i] = 0;
            for (int j = 0; j < output_layer.size(); ++j) {
                hidden_error[i] += output_error[j] * hidden_output_weights[j * hidden_layer.size() + i];
            }
            hidden_error[i] *= sigmoid_derivative(hidden_layer[i]);
        }

        // Aktualizacja wag (gradient descent)
        for (int i = 0; i < hidden_output_weights.size(); ++i) {
            hidden_output_weights[i] += learning_rate * output_error[i % output_layer.size()] * hidden_layer[i / output_layer.size()];
        }
        for (int i = 0; i < input_hidden_weights.size(); ++i) {
            input_hidden_weights[i] += learning_rate * hidden_error[i % hidden_layer.size()] * inputs[i / hidden_layer.size()];
        }
    }

private:
    std::vector<double> input_hidden_weights;
    std::vector<double> hidden_output_weights;
    std::vector<double> hidden_layer;
    std::vector<double> output_layer;
};

/////////////////////////////////////////////////
int main() {
    // Inicjalizacja sieci neuronowej
    SimpleNN nn(100, 15, 1); // 100 wejść (10x10), 15 neuronów ukrytych, 1 wyjście

    // Obrazy treningowe i oczekiwane wyniki
    std::vector<std::string> training_files = {"training1.txt", "training2.txt", "training3.txt"};
    std::vector<int> expected_outputs = {1, 0, 0}; // Oczekiwane wyjścia dla obrazów (np. 1 oznacza obecność kształtu)

    // Trenowanie na obrazach
    for (int epoch = 0; epoch < 10000; ++epoch) {
        for (int i = 0; i < training_files.size(); ++i) {
            std::vector<int> image = readImage(training_files[i]);
            std::vector<int> expected_output = {expected_outputs[i]}; // Jedno wyjście per obraz
            nn.train(image, expected_output, 0.01); // Trenuj na każdym obrazie
        }
    }

    // Testowanie na obrazie "obraz.txt"
    std::vector<int> test_image = readImage("detection.txt");
    std::vector<double> result = nn.forward(test_image);

    // Wyświetlenie wyniku
    std::cout << "Wynik: " << result[0] << std::endl; // Powinno być blisko 1 lub 0, w zależności od rozpoznanego wzoru

    return 0;
}
