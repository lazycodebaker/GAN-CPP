#ifndef _CONV_BLOCK
#define _CONV_BLOCK

#include "include.cpp"

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

#endif