#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <cstdlib>

void load_matrix(const std::string& filename, std::vector<double>& data, int64_t& rows, int64_t& cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file\n";
        throw std::runtime_error("Cannot open file");
    }

    file.read(reinterpret_cast<char*>(&rows), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int64_t));

    data.resize(rows * cols);
    file.read(reinterpret_cast<char*>(data.data()), rows * cols * sizeof(double));
}

int main(int argc, char* argv[]) {
    std::vector<std::string> datasets;
    std::vector<size_t> instance_sizes;
    int restarts = 1;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--datasets") {
            ++i;
            while (i < argc && argv[i][0] != '-') {
                datasets.push_back(argv[i]);
                ++i;
            }
            --i;
        } else if (arg == "--instance_sizes") {
            ++i;
            while (i < argc && argv[i][0] != '-') {
                instance_sizes.push_back(std::stoul(argv[i]));
                ++i;
            }
            --i;
        } else if (arg == "--restarts") {
            if (i + 1 < argc) {
                restarts = std::stoi(argv[++i]);
            }
        }
    }

    int64_t rows, cols;
    std::vector<double> data;

    for (const auto& dataset : datasets) {
        for (const auto& instance_size : instance_sizes) {
            // generate the matrix using Julia script
            std::cout << "DATASET '" << dataset << "' WITH INSTANCE SIZE " << instance_size << std::endl;

            // load the matrix from the generated binary file
            std::string filename = "data/bin/" + dataset + "-" + std::to_string(instance_size) + ".bin";
            load_matrix(filename, data, rows, cols);

            for (int r = 0; r < restarts; ++r) {
                
            }
        }
    }

    return 0;
}