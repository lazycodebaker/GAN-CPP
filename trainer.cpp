
#ifndef _TRAINER
#define _TRAINER

#define EPOCHS 10
#define FAKE_LABEL 0.0
#define REAL_LABEL 1.0

#include "include.cpp"
#include "read_mnist_images.cpp"
#include "read_mnist_labels.cpp"
#include "generator.cpp"
#include "discriminator.cpp"

int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};

class GANTrainer
{
public:
    GANTrainer(const std::string &image_file, const std::string &label_file, const std::string &root)
        : device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU), root(root)
    {
        std::cout << "Using device: " << device << std::endl;
        images = readMNISTImages(image_file);
        labels = readMNISTLabels(label_file);
    }

    void train()
    {
        if (images.empty())
        {
            std::cerr << "Failed to load images." << std::endl;
            return;
        }

        torch::Tensor fixed_noise = torch::randn({64, 100, 1, 1});
        Generator G = create_generator();
        Discriminator D = create_discriminator();
        G.apply(weights_init);
        D.apply(weights_init);

        torch::optim::Adam optim_G(G.parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.999)));
        torch::optim::Adam optim_D(D.parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.999)));
        auto criterion = torch::nn::BCELoss();

        auto train_dataset = MNISTDataset(root, true).map(torch::data::transforms::Stack<>());
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_dataset), 64);

        for (int epoch = 1; epoch <= EPOCHS; ++epoch)
        {
            float loss_d = 0.0, loss_g = 0.0;
            int num_batches = 0;

            for (auto &batch : *data_loader)
            {
                auto images = batch.data.to(device);
                auto real_labels = torch::full({images.size(0)}, REAL_LABEL, device);
                auto fake_labels = torch::full({images.size(0)}, FAKE_LABEL, device);

                // Train Discriminator
                D.zero_grad();
                auto d_real = D.forward(images).view(-1);
                auto d_loss_real = criterion(d_real, real_labels);
                d_loss_real.backward();

                auto noise = torch::randn({images.size(0), 100, 1, 1}, device);
                auto fake_images = G.forward(noise);
                auto d_fake = D.forward(fake_images.detach()).view(-1);
                auto d_loss_fake = criterion(d_fake, fake_labels);
                d_loss_fake.backward();
                optim_D.step();

                auto d_loss = d_loss_real + d_loss_fake;
                loss_d += d_loss.item<float>();

                // Train Generator
                G.zero_grad();
                auto d_fake_gen = D.forward(fake_images).view(-1);
                auto g_loss = criterion(d_fake_gen, real_labels);
                g_loss.backward();
                optim_G.step();

                loss_g += g_loss.item<float>();
                num_batches++;
            }

            std::cout << "Epoch [" << epoch << "/" << EPOCHS << "], "
                      << "Loss_D: " << (loss_d / num_batches) << ", "
                      << "Loss_G: " << (loss_g / num_batches) << std::endl;
        }
    }

private:
    torch::Device device;
    std::vector<std::vector<unsigned char>> images;
    std::vector<unsigned char> labels;
    std::string root;
};

#endif