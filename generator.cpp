
#ifndef _GENERATOR
#define _GENERATOR

#include "include.cpp"
#include "disc_conv.cpp"

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


#endif