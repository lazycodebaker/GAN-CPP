#ifndef _CUSTOM_SEQUENTIAL_IMPL
#define _CUSTOM_SEQUENTIAL_IMPL

#include "torch/torch.h"

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

#endif 