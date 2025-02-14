
#include "include.cpp"
#include "trainer.cpp"

int main()
{
    std::string image_file = "/Users/anshumantiwari/Documents/codes/personal/C++/GAN/MNIST_data/train-images-idx3-ubyte";
    std::string label_file = "/Users/anshumantiwari/Documents/codes/personal/C++/GAN/MNIST_data/train-labels-idx1-ubyte";
    std::string root = "/Users/anshumantiwari/Documents/codes/personal/C++/GAN/MNIST_data";

    GANTrainer trainer(image_file, label_file, root);
    trainer.train();
    return 0;
}
