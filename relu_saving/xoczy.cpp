#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

// Funkcja do odczytu obrazu z pliku tekstowego i normalizacji wartości pikseli (0 lub 1)
std::vector<double> readImage(const std::string& filename) {
    std::ifstream file(filename);  // Otwiera plik z obrazem
    std::vector<double> image;     // Wektor przechowujący wartości pikseli
    if (file.is_open()) {          // Sprawdza, czy plik został poprawnie otwarty
        std::string line;
        // Odczytuje każdą linię z pliku
        while (std::getline(file, line)) {
            for (char pixel : line) {
                // Konwertuje znaki '0' i '1' na wartości 0.0 i 1.0, aby uzyskać normalizowane wartości pikseli
                if (pixel == '0' || pixel == '1') {
                    image.push_back((double)(pixel - '0'));  // Zamienia '0' lub '1' na double
                }
            }
        }
        file.close();  // Zamknięcie pliku
    }
    return image;  // Zwraca wektor pikseli (100 elementów, bo obraz ma rozmiar 10x10)
}

// Prosta sieć neuronowa z jedną warstwą ukrytą oraz funkcjami aktywacji ReLU i Sigmoid
class SimpleNN {
public:
    // Konstruktor inicjalizujący sieć neuronową
    SimpleNN(int input_size, int hidden_size, int output_size) {
        // Inicjalizacja wektorów wag dla połączeń między warstwami
        input_hidden_weights.resize(input_size * hidden_size);  // Wagi między warstwą wejściową a ukrytą
        hidden_output_weights.resize(hidden_size * output_size); // Wagi między warstwą ukrytą a wyjściową
        hidden_layer.resize(hidden_size);  // Warstwa ukryta (30 neuronów)
        output_layer.resize(output_size);  // Warstwa wyjściowa (1 neuron)

        // Inicjalizacja wag losowymi wartościami z zakresu [-1, 1]
        for (double& weight : input_hidden_weights) {
            weight = ((double) rand() / RAND_MAX) * 2 - 1;
        }
        for (double& weight : hidden_output_weights) {
            weight = ((double) rand() / RAND_MAX) * 2 - 1;
        }
    }

    // Funkcja forward propagation (przepływ sygnału od wejścia do wyjścia)
    std::vector<double> forward(const std::vector<double>& inputs) {
        // Obliczanie wartości neuronów w warstwie ukrytej
        for (int i = 0; i < hidden_layer.size(); ++i) {
            hidden_layer[i] = 0;
            for (int j = 0; j < inputs.size(); ++j) {
                hidden_layer[i] += inputs[j] * input_hidden_weights[i * inputs.size() + j]; // Suma ważona
            }
            hidden_layer[i] = relu(hidden_layer[i]);  // Aktywacja ReLU
        }

        // Obliczanie wartości neuronów w warstwie wyjściowej
        for (int i = 0; i < output_layer.size(); ++i) {
            output_layer[i] = 0;
            for (int j = 0; j < hidden_layer.size(); ++j) {
                output_layer[i] += hidden_layer[j] * hidden_output_weights[i * hidden_layer.size() + j]; // Suma ważona
            }
            output_layer[i] = sigmoid(output_layer[i]);  // Aktywacja sigmoid
        }

        return output_layer;  // Zwraca wartości neuronów w warstwie wyjściowej
    }

    // Funkcja uczenia sieci neuronowej (backpropagation)
    void train(const std::vector<double>& inputs, const std::vector<double>& expected_output, double learning_rate) {
        forward(inputs);  // Przeprowadza forward propagation na wejściu

        // Obliczanie błędów dla warstwy wyjściowej
        std::vector<double> output_error(output_layer.size());
        for (int i = 0; i < output_layer.size(); ++i) {
            output_error[i] = expected_output[i] - output_layer[i];  // Różnica między oczekiwanym a rzeczywistym wynikiem
        }

        // Obliczanie błędów dla warstwy ukrytej
        std::vector<double> hidden_error(hidden_layer.size());
        for (int i = 0; i < hidden_layer.size(); ++i) {
            hidden_error[i] = 0;
            for (int j = 0; j < output_layer.size(); ++j) {
                hidden_error[i] += output_error[j] * hidden_output_weights[j * hidden_layer.size() + i];  // Błąd sumowany dla każdego wyjścia
            }
            hidden_error[i] *= relu_derivative(hidden_layer[i]);  // Mnożenie przez pochodną ReLU
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

    // Funkcja zapisująca wagi do pliku
    void saveWeights(const std::string& filename) {
        std::ofstream file(filename);
        if (file.is_open()) {
            for (const double& weight : input_hidden_weights) {
                file << weight << "\n";  // Zapis wag między wejściem a ukrytą warstwą
            }
            for (const double& weight : hidden_output_weights) {
                file << weight << "\n";  // Zapis wag między ukrytą a wyjściową warstwą
            }
            file.close();
        } else {
            std::cerr << "Could not open file for writing." << std::endl;
        }
    }

    // Funkcja wczytująca wagi z pliku
    void loadWeights(const std::string& filename) {
        std::ifstream file(filename);
        if (file.is_open()) {
            for (double& weight : input_hidden_weights) {
                file >> weight;  // Wczytywanie wag między wejściem a warstwą ukrytą
            }
            for (double& weight : hidden_output_weights) {
                file >> weight;  // Wczytywanie wag między ukrytą a wyjściową warstwą
            }
            file.close();
        } else {
            std::cerr << "Could not open file for reading." << std::endl;
        }
    }

private:
    std::vector<double> input_hidden_weights;  // Wagi między warstwą wejściową a ukrytą
    std::vector<double> hidden_output_weights;  // Wagi między warstwą ukrytą a wyjściową
    std::vector<double> hidden_layer;  // Neurony w warstwie ukrytej
    std::vector<double> output_layer;  // Neurony w warstwie wyjściowej

    // Funkcja aktywacji sigmoid
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Pochodna funkcji sigmoid (używana w backpropagation)
    double sigmoid_derivative(double x) {
        return x * (1.0 - x);
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
    // Inicjalizacja sieci neuronowej: 100 neuronów wejściowych, 30 ukrytych i 1 wyjściowy
    SimpleNN nn(100, 30, 1);

    std::cout << "Czy chcesz uczyć sieć? (1 - tak, 0 - nie): ";
    int choice;
    std::cin >> choice;

    if (choice == 1) {
        // Obrazy treningowe i oczekiwane wyniki dla sieci (1 dla X, 0 dla innych)
        std::vector<std::string> training_files = {"training1.txt", "training2.txt", "training3.txt"};
        std::vector<int> expected_outputs = {1, 0, 0};  // Trenuj sieć: X = 1, inne = 0

        // Trenowanie sieci przez 20 000 epok
        for (int epoch = 0; epoch < 20000; ++epoch) {
            for (int i = 0; i < training_files.size(); ++i) {
                std::vector<double> image = readImage(training_files[i]);  // Odczyt obrazu
                std::vector<double> expected_output = {static_cast<double>(expected_outputs[i])};  // Oczekiwany wynik
                nn.train(image, expected_output, 0.01);  // Uczenie sieci z learning rate 0.01
            }
        }

        // Zapisz wagi do pliku
        nn.saveWeights("weights.txt");
        std::cout << "Sieć została wytrenowana i wagi zapisane do weights.txt." << std::endl;

    } else {
        // Wczytaj wagi z pliku, jeśli sieć już była uczona
        nn.loadWeights("weights.txt");
        std::cout << "Wagi wczytane z weights.txt." << std::endl;

        // Testowanie sieci na nowym obrazie "detection.txt"
        std::vector<double> test_image = readImage("detection.txt");  // Odczyt obrazu testowego
        std::vector<double> result = nn.forward(test_image);  // Obliczenie wyniku sieci

        // Wyświetlenie wyniku (powinien być bliski 1 dla X)
        std::cout << "Wynik: " << result[0] << std::endl;
    }

    return 0;
}

