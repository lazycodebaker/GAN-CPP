
#ifndef _READ_MNIST_IMAGES
#define _READ_MNIST_IMAGES

#include "include.cpp"

std::vector<std::vector<unsigned char>> readMNISTImages(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return std::vector<std::vector<unsigned char>>();
    }

    int magic_number = 0, num_images = 0, rows = 0, cols = 0;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
    num_images = reverseInt(num_images);
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    rows = reverseInt(rows);
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    cols = reverseInt(cols);

    std::cout << "MNIST Image Dataset Info:" << std::endl;
    std::cout << "Magic Number: " << magic_number << std::endl;
    std::cout << "Number of Images: " << num_images << std::endl;
    std::cout << "Image Dimensions: " << rows << " x " << cols << std::endl;

    std::vector<std::vector<unsigned char>> images(num_images, std::vector<unsigned char>(rows * cols));

    for (int i = 0; i < num_images; i++)
    {
        file.read(reinterpret_cast<char *>(images[i].data()), rows * cols);
    };

    file.close();
    return images;
};

// Function to read MNIST images
torch::Tensor read_mnist_images(const std::string &path, bool train)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open file: " + path);
    }

    int magic_number = 0, num_images = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

    // Reverse the byte order if necessary (MNIST data is in big-endian format)
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    auto tensor = torch::empty({num_images, 1, rows, cols}, torch::kByte);
    file.read(reinterpret_cast<char *>(tensor.data_ptr()), tensor.numel());

    return tensor.to(torch::kFloat32).div_(255.0);
}

#endif