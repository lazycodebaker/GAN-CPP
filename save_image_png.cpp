
#ifndef _SAVE_IMAGES_PNG
#define _SAVE_IMAGES_PNG

#include "include.cpp"

void saveImagesAsPNG(const std::vector<std::vector<unsigned char>> &images, int rows, int cols)
{
    for (int i = 0; i < 5 && i < images.size(); i++)
    {
        sf::Image image = sf::Image(sf::Vector2u(cols, rows));

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                unsigned char pixelValue = images[i][y * cols + x];
                image.setPixel(sf::Vector2u(x, y), sf::Color(pixelValue, pixelValue, pixelValue));
            }
        }

        std::string filename = "mnist_image_" + std::to_string(i) + ".png";
        if (image.saveToFile(filename))
        {
            std::cout << "Saved: " << filename << std::endl;
        }
        else
        {
            std::cerr << "Failed to save: " << filename << std::endl;
        }
    }
};

#endif