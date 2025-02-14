
#ifndef _INIT_WEIGHT
#define _INIT_WEIGHT

#include "include.cpp"

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

#endif