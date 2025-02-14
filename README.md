# GAN for MNIST Dataset

This project implements a Generative Adversarial Network (GAN) to generate images similar to those in the MNIST dataset. The GAN consists of two neural networks: a Generator and a Discriminator, which are trained simultaneously in a competitive manner. The Generator learns to produce realistic images, while the Discriminator learns to distinguish between real and fake images.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Project Structure](#project-structure)
4. [Building the Project](#building-the-project)
5. [Training the GAN](#training-the-gan)
6. [Saving and Displaying Images](#saving-and-displaying-images)
7. [Custom Sequential Module](#custom-sequential-module)
8. [License](#license)

---

## Project Overview

The goal of this project is to train a GAN to generate realistic images of handwritten digits from the MNIST dataset. The GAN is implemented using the **LibTorch** library, which provides a C++ frontend for PyTorch. The project also uses **SFML** for image display and **OpenCV** for additional image processing capabilities (if needed).

### Key Features:
- **Generator**: A neural network that generates images from random noise.
- **Discriminator**: A neural network that distinguishes between real and fake images.
- **Training Loop**: Alternates between training the Discriminator and the Generator.
- **MNIST Dataset**: Loads and preprocesses the MNIST dataset for training.
- **Image Visualization**: Uses SFML to display and save generated images.

---

## Dependencies

The project relies on the following libraries:
- **LibTorch**: C++ frontend for PyTorch. Download and install from the [official PyTorch website](https://pytorch.org/).
- **SFML**: Simple and Fast Multimedia Library for image display. Install via Homebrew or download from the [SFML website](https://www.sfml-dev.org/).
- **OpenCV**: Optional for additional image processing. Install via Homebrew or from the [OpenCV website](https://opencv.org/).

### Installation Instructions:
1. **LibTorch**:
   - Download the C++ distribution of LibTorch.
   - Set the `Torch_DIR` in the `CMakeLists.txt` file to point to the LibTorch installation directory.

2. **SFML**:
   - Install SFML using Homebrew:
     ```bash
     brew install sfml
     ```
   - Ensure the `SFML_DIR` in `CMakeLists.txt` points to the correct installation path.

3. **OpenCV**:
   - Install OpenCV using Homebrew:
     ```bash
     brew install opencv
     ```
   - Update the `OpenCV_DIR` in `CMakeLists.txt` if necessary.

---

## Project Structure

The project is organized as follows:
- **`include.cpp`**: Contains common includes and definitions.
- **`trainer.cpp`**: Implements the `GANTrainer` class, which handles the training loop.
- **`generator.cpp`**: Defines the Generator model.
- **`discriminator.cpp`**: Defines the Discriminator model.
- **`read_mnist_images.cpp`**: Functions to load MNIST images.
- **`read_mnist_labels.cpp`**: Functions to load MNIST labels.
- **`save_images_png.cpp`**: Functions to save generated images as PNG files.
- **`display_images.cpp`**: Functions to display images using SFML.
- **`dataset.cpp`**: Implements the `MNISTDataset` class for data loading.
- **`conv_block.cpp`**: Utility functions for creating convolutional blocks.
- **`disc_conv.cpp`**: Utility functions for creating Discriminator convolutional layers.
- **`custom_sequential.cpp`**: Implements a custom sequential module for neural networks.
- **`CMakeLists.txt`**: CMake configuration file for building the project.

---

## Building the Project

To build the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/lazycodebaker/GAN-CPP.git
   cd gan-mnist
   ```

2. Create a build directory and compile the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Run the executable:
   ```bash
   ./gan
   ```

---

## Training the GAN

The training process is handled by the `GANTrainer` class. The main steps include:
1. Loading the MNIST dataset.
2. Initializing the Generator and Discriminator models.
3. Alternating between training the Discriminator and the Generator.
4. Evaluating the loss functions and updating the models.

### Training Loop:
- The Discriminator is trained to maximize the probability of correctly classifying real and fake images.
- The Generator is trained to minimize the probability of the Discriminator correctly classifying its output as fake.

### Hyperparameters:
- **Learning Rate**: 2e-4 for both Generator and Discriminator.
- **Batch Size**: 64.
- **Epochs**: 10.
- **Optimizer**: Adam with betas (0.5, 0.999).

---

## Saving and Displaying Images

The project includes utilities to save and display generated images:
- **`saveImagesAsPNG`**: Saves the first few generated images as PNG files.
- **`displayImages`**: Displays the generated images in an SFML window.

---

## Custom Sequential Module

The `CustomSequential` module is a wrapper around `torch::nn::Sequential` that allows for more flexible model definitions. It is used in both the Generator and Discriminator models.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- The MNIST dataset is provided by Yann LeCun and Corinna Cortes.
- This project is inspired by the PyTorch GAN tutorial and adapted for C++ using LibTorch.

---

For any questions or issues, please open an issue on the GitHub repository. Contributions are welcome!
