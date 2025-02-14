#ifndef _DATASET
#define _DATASET

#include "include.cpp"

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

#endif