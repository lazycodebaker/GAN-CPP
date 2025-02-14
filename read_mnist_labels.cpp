
#ifndef _READ_MNIST_LABELS
#define _READ_MNIST_LABELS

#include "include.cpp"

std::vector<unsigned char> readMNISTLabels(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return std::vector<unsigned char>();
    };

    int magic_number = 0, num_labels = 0;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));
    num_labels = reverseInt(num_labels);

    std::cout << "MNIST Label Dataset Info:" << std::endl;
    std::cout << "Magic Number: " << magic_number << std::endl;
    std::cout << "Number of Labels: " << num_labels << std::endl;

    std::vector<unsigned char> labels(num_labels);
    file.read(reinterpret_cast<char *>(labels.data()), num_labels);

    file.close();
    return labels;
};

torch::Tensor read_mnist_labels(const std::string &path, bool train)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open file: " + path);
    }

    int magic_number = 0, num_labels = 0;
    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));

    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    auto tensor = torch::empty(num_labels, torch::kByte);
    file.read(reinterpret_cast<char *>(tensor.data_ptr()), num_labels);

    return tensor.to(torch::kInt64);
};

#endif