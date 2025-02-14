#ifndef _DISC_CONV
#define _DISC_CONV

#include "include.cpp"

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

#endif