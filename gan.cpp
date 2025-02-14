#include "iostream"
#include "fstream"
#include "vector"
#include "cmath"
#include "algorithm"
#include "string"
#include "SFML/Graphics.hpp"
#include "SFML/Window.hpp"
#include "SFML/System.hpp"
#include "SFML/Window/VideoMode.hpp"
#include "SFML/Graphics/Sprite.hpp"
#include "SFML/Graphics/Image.hpp"
#include "SFML/Window/Window.hpp"
#include "torch/torch.h"

#define EPOCHS 10
#define FAKE_LABEL 0.0
#define REAL_LABEL 1.0

struct CustomSequentialImpl : torch::nn::Module
{
    torch::nn::Sequential seq;

    CustomSequentialImpl() : seq(nullptr) {};

    CustomSequentialImpl(torch::nn::Sequential sequential) : seq(sequential)
    {
        register_module("seq", seq);
    };

    torch::Tensor forward(torch::Tensor x)
    {
        return seq->forward(x);
    };
};

TORCH_MODULE(CustomSequential);

int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

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
}

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

void saveImagesAsPNG(const std::vector<std::vector<unsigned char>> &images, int rows, int cols)
{
    for (int i = 0; i < 5 && i < images.size(); i++)
    {
        sf::Image image = sf::Image(sf::Vector2u(cols, rows));

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                unsigned char pixelValue = images[i][y * cols + x];
                image.setPixel(sf::Vector2u(x, y), sf::Color(pixelValue, pixelValue, pixelValue));
            }
        }

        std::string filename = "mnist_image_" + std::to_string(i) + ".png";
        if (image.saveToFile(filename))
        {
            std::cout << "Saved: " << filename << std::endl;
        }
        else
        {
            std::cerr << "Failed to save: " << filename << std::endl;
        }
    }
};

void displayImages(const std::vector<std::vector<unsigned char>> &images, int rows, int cols)
{
    sf::VideoMode video_mode(sf::Vector2u(cols, rows));
    sf::RenderWindow window(video_mode, "MNIST Images");

    std::vector<sf::Texture> textures;
    std::vector<sf::Sprite> sprites;

    for (int i = 0; i < 5 && i < images.size(); i++)
    {
        sf::Image img = sf::Image(sf::Vector2u(cols, rows));
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                unsigned char pixelValue = images[i][y * cols + x];
                img.setPixel(sf::Vector2u(x, y), sf::Color(pixelValue, pixelValue, pixelValue));
            }
        }

        sf::Texture texture;
        if (!textures[i].loadFromImage(img))
        {
            std::cerr << "Failed to load image into texture at index " << i << std::endl;
        };
        textures.push_back(std::move(texture));

        sf::Sprite sprite = sf::Sprite(texture);
        sprite.setTexture(textures.back());
        sprite.setPosition(sf::Vector2f(static_cast<float>(i * cols), 0));
        sprites.push_back(std::move(sprite));
    };

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
        };

        window.clear();
        for (const auto &sprite : sprites)
            window.draw(sprite);
        window.display();
    };
};

void weights_init(torch::nn::Module &m)
{
    std::string classname = typeid(m).name();

    if (classname.find("Conv") != std::string::npos)
    {
        if (auto *conv = dynamic_cast<torch::nn::Conv2dImpl *>(&m))
        {
            torch::nn::init::normal_(conv->weight, 0.0, 0.02);
            if (conv->bias.defined())
            {
                torch::nn::init::constant_(conv->bias, 0);
            };
        };
    }
    else if (classname.find("BatchNorm") != std::string::npos)
    {
        if (auto *bn = dynamic_cast<torch::nn::BatchNorm2dImpl *>(&m))
        {
            torch::nn::init::normal_(bn->weight, 1.0, 0.02);
            torch::nn::init::constant_(bn->bias, 0);
        };
    };
};

torch::nn::Sequential disc_conv(int in_c, int out_c, int ks = 4, int stride = 2, int padding = 1, bool bn = true, bool out_layer = false)
{
    torch::nn::Sequential layers;
    layers->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_c, out_c, ks).stride(stride).padding(padding).bias(false)));

    if (bn)
    {
        layers->push_back(torch::nn::BatchNorm2d(out_c));
    };
    if (out_layer)
    {
        layers->push_back(torch::nn::Sigmoid());
    }
    else
    {
        layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
    };
    return layers;
};

// Define the Discriminator model
struct Discriminator : torch::nn::Module
{
    CustomSequential conv_layers;

    Discriminator()
    {
        conv_layers = CustomSequential(torch::nn::Sequential(
            disc_conv(1, 32, false),                // First convolution layer
            disc_conv(32, 64),                      // Second convolution layer
            disc_conv(64, 128, 3),                  // Third convolution layer
            disc_conv(128, 1, 4, 1, 0, false, true) // Final convolution layer (output layer)
            ));

        register_module("conv_layers", conv_layers);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return conv_layers->forward(x);
    }
};

// Function to create and return the Discriminator model
Discriminator create_discriminator()
{
    return Discriminator();
};

torch::nn::Sequential conv_block(int in_c, int out_c, int ks = 4, bool out_layer = false, int stride = 1, int padding = 1, bool bn = true)
{
    torch::nn::Sequential block;

    // Add ConvTranspose2d layer
    block->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(in_c, out_c, ks).stride(stride).padding(padding).bias(false)));

    // BatchNorm layer
    if (bn)
    {
        block->push_back(torch::nn::BatchNorm2d(out_c));
    }

    // Output layer: Tanh if it's the final layer, otherwise ReLU
    if (out_layer)
    {
        block->push_back(torch::nn::Tanh());
    }
    else
    {
        block->push_back(torch::nn::ReLU(true));
    }

    return block;
};

struct Generator : torch::nn::Module
{
    CustomSequential layers;

    Generator()
    {
        layers = CustomSequential(torch::nn::Sequential(
            conv_block(100, 128, 0, false),   // First convolution layer
            conv_block(128, 64, 3, true),     // Second convolution layer
            conv_block(64, 32, 1, true),      // Third convolution layer
            conv_block(32, 1, 2, false, true) // Final convolution layer (output layer)
            ));

        register_module("layers", layers);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return layers->forward(x);
    }

    /*
    // Convolution block
    torch::nn::Sequential conv_block(int in_c, int out_c, int ks = 4, bool out_layer = false, int stride = 1, int padding = 1, bool bn = true)
    {
        torch::nn::Sequential block;

        // Add ConvTranspose2d layer
        block->push_back(torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(in_c, out_c, ks).stride(stride).padding(padding).bias(false)));

        // BatchNorm layer
        if (bn)
        {
            block->push_back(torch::nn::BatchNorm2d(out_c));
        }

        // Output layer: Tanh if it's the final layer, otherwise ReLU
        if (out_layer)
        {
            block->push_back(torch::nn::Tanh());
        }
        else
        {
            block->push_back(torch::nn::ReLU(true));
        }

        return block;
    }
        */
};

// Function to create and return the Generator model
Generator create_generator()
{
    return Generator();
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

// Function to read MNIST labels
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

    // Reverse the byte order if necessary
    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    auto tensor = torch::empty(num_labels, torch::kByte);
    file.read(reinterpret_cast<char *>(tensor.data_ptr()), num_labels);

    return tensor.to(torch::kInt64);
};

class MNISTDataset : public torch::data::Dataset<MNISTDataset>
{
public:
    MNISTDataset(const std::string &root, bool train)
    {
        images_ = read_mnist_images(root + (train ? "/train-images-idx3-ubyte" : "/t10k-images-idx3-ubyte"), train);
        labels_ = read_mnist_labels(root + (train ? "/train-labels-idx1-ubyte" : "/t10k-labels-idx1-ubyte"), train);
    }

    // Override the get method to return a single example
    torch::data::Example<> get(size_t index) override
    {
        return {images_[index], labels_[index]};
    }

    // Override the size method to return the size of the dataset
    torch::optional<size_t> size() const override
    {
        return images_.size(0);
    }

private:
    torch::Tensor images_, labels_;
};

int main()
{
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    std::string image_file = "/Users/anshumantiwari/Documents/codes/personal/C++/GAN/MNIST_data/train-images-idx3-ubyte";
    std::string label_file = "/Users/anshumantiwari/Documents/codes/personal/C++/GAN/MNIST_data/train-labels-idx1-ubyte";

    std::vector<std::vector<unsigned char>> images = readMNISTImages(image_file);
    std::vector<unsigned char> labels = readMNISTLabels(label_file);

    if (!images.empty())
    {
        torch::Tensor fixed_noise = torch::randn({64, 100, 1, 1});

        Generator G = create_generator();
        Discriminator D = create_discriminator();

        G.apply(weights_init);
        D.apply(weights_init);

        // Define optimizers
        torch::optim::Adam optim_G(G.parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.999)));
        torch::optim::Adam optim_D(D.parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.999)));

        // Define loss function
        auto criterion = torch::nn::BCELoss();

        std::string root = "/Users/anshumantiwari/Documents/codes/personal/C++/GAN/MNIST_data";

        // Create the dataset
        auto train_dataset = MNISTDataset(root, true).map(torch::data::transforms::Stack<>());

        // Create a data loader
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_dataset), 64);

        // Training loop
        for (int epoch = 1; epoch <= EPOCHS; ++epoch)
        {
            float loss_d = 0.0;
            float loss_g = 0.0;
            int num_batches = 0; // Counter for the number of batches

            for (auto &batch : *data_loader)
            {
                auto data = batch.data;
                auto target = batch.target;
                std::cout << "Batch data size: " << data.sizes() << std::endl;
                std::cout << "Batch target size: " << target.sizes() << std::endl;

                auto images = batch.data.to(device);
                auto real_labels = torch::full({images.size(0)}, REAL_LABEL, device);
                auto fake_labels = torch::full({images.size(0)}, FAKE_LABEL, device);

                // Train Discriminator
                D.zero_grad();

                // Real images
                auto d_real = D.forward(images).view(-1);
                auto d_loss_real = criterion(d_real, real_labels);
                d_loss_real.backward();

                // Fake images
                auto noise = torch::randn({images.size(0), 100, 1, 1}, device);
                auto fake_images = G.forward(noise);
                auto d_fake = D.forward(fake_images.detach()).view(-1);
                auto d_loss_fake = criterion(d_fake, fake_labels);
                d_loss_fake.backward();

                auto d_loss = d_loss_real + d_loss_fake;
                optim_D.step();

                // Train Generator
                G.zero_grad();

                auto d_fake_gen = D.forward(fake_images).view(-1);
                auto g_loss = criterion(d_fake_gen, real_labels);
                g_loss.backward();
                optim_G.step();

                // Accumulate losses
                loss_d += d_loss.item<float>();
                loss_g += g_loss.item<float>();
                num_batches++; // Increment batch counter
            }

            // Calculate average losses
            float avg_loss_d = loss_d / num_batches;
            float avg_loss_g = loss_g / num_batches;

            std::cout << "Epoch [" << epoch << "/" << EPOCHS << "], "
                      << "Loss_D: " << avg_loss_d << ", "
                      << "Loss_G: " << avg_loss_g << std::endl;
        }
    }
    return 0;
}