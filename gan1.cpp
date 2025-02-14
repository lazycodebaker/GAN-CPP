#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>

// Function to initialize weights
void weights_init(torch::nn::Module& module) {
    for (auto& m : module.children()) {
        if (auto* conv = dynamic_cast<torch::nn::Conv2dImpl*>(m.get())) {
            torch::nn::init::normal_(conv->weight, 0.0, 0.02);
        } else if (auto* batch_norm = dynamic_cast<torch::nn::BatchNorm2dImpl*>(m.get())) {
            torch::nn::init::normal_(batch_norm->weight, 1.0, 0.02);
            torch::nn::init::constant_(batch_norm->bias, 0);
        }
    }
}

// Discriminator class
struct Discriminator : torch::nn::Module {
    torch::nn::Sequential model;

    Discriminator() {
        model = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 4).stride(2).padding(1)),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 4).stride(2).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 1, 4).stride(1).padding(0)),
            torch::nn::Sigmoid()
        );
        register_module("model", model);
    }

    torch::Tensor forward(torch::Tensor x) {
        return model->forward(x);
    }
};

// Generator class
struct Generator : torch::nn::Module {
    torch::nn::Sequential model;

    Generator() {
        model = torch::nn::Sequential(
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(100, 128, 4).stride(1).padding(0)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 4).stride(2).padding(1)),
            torch::nn::BatchNorm2d(32),
            torch::nn::ReLU(),
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(32, 1, 4).stride(2).padding(1)),
            torch::nn::Tanh()
        );
        register_module("model", model);
    }

    torch::Tensor forward(torch::Tensor x) {
        return model->forward(x);
    }
};

void visualize(torch::Tensor image_tensor) {
    image_tensor = image_tensor.squeeze().detach().cpu();
    sf::Image image;
    image.create(28, 28);
    
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            uint8_t pixel_value = static_cast<uint8_t>((image_tensor[y][x].item<float>() + 1) * 127.5);
            image.setPixel(x, y, sf::Color(pixel_value, pixel_value, pixel_value));
        }
    }
    
    sf::Texture texture;
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);
    sf::RenderWindow window(sf::VideoMode(280, 280), "Generated Image");
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
        }
        window.clear();
        window.draw(sprite);
        window.display();
    }
}

int main() {
    torch::Device device(torch::kCUDA);

    // Load MNIST dataset
    auto dataset = torch::data::datasets::MNIST("./MNIST_data")
                        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                        .map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(std::move(dataset), 128);

    Discriminator D;
    Generator G;
    D.to(device);
    G.to(device);
    
    weights_init(D);
    weights_init(G);
    
    torch::optim::Adam optim_D(D.parameters(), torch::optim::AdamOptions(0.0002).betas({0.5, 0.999}));
    torch::optim::Adam optim_G(G.parameters(), torch::optim::AdamOptions(0.0002).betas({0.5, 0.999}));
    
    const int epochs = 10;
    const float real_label = 1.0, fake_label = 0.0;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (auto& batch : *data_loader) {
            auto images = batch.data.to(device);
            auto real_labels = torch::full({images.size(0)}, real_label, torch::kCUDA);
            auto fake_labels = torch::full({images.size(0)}, fake_label, torch::kCUDA);

            // Train Discriminator
            optim_D.zero_grad();
            auto d_real = D.forward(images).view(-1);
            auto d_loss_real = torch::binary_cross_entropy(d_real, real_labels);
            d_loss_real.backward();

            auto noise = torch::randn({images.size(0), 100, 1, 1}, device);
            auto fake_images = G.forward(noise);
            auto d_fake = D.forward(fake_images.detach()).view(-1);
            auto d_loss_fake = torch::binary_cross_entropy(d_fake, fake_labels);
            d_loss_fake.backward();

            optim_D.step();

            // Train Generator
            optim_G.zero_grad();
            d_fake = D.forward(fake_images).view(-1);
            auto g_loss = torch::binary_cross_entropy(d_fake, real_labels);
            g_loss.backward();
            optim_G.step();
        }
    }
    
    auto noise = torch::randn({1, 100, 1, 1}, device);
    auto generated_image = G.forward(noise);
    visualize(generated_image);
    
    return 0;
}
