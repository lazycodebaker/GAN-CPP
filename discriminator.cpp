
#ifndef _DISCRIMINATOR
#define _DISCRIMINATOR

#include "include.cpp"
#include "disc_conv.cpp"

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

#endif