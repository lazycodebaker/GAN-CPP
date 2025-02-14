

#ifndef _DISPLAY_IMAGES
#define _DISPLAY_IMAGES

#include "include.cpp"

void displayImages(const std::vector<std::vector<unsigned char>> &images, int rows, int cols)
{
    sf::VideoMode video_mode(sf::Vector2u(cols, rows));
    sf::RenderWindow window(video_mode, "MNIST Images");

    std::vector<sf::Texture> textures;
    std::vector<sf::Sprite> sprites;

    for (int i = 0; i < 5 && i < images.size(); i++)
    {
        sf::Image img = sf::Image(sf::Vector2u(cols, rows));
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                unsigned char pixelValue = images[i][y * cols + x];
                img.setPixel(sf::Vector2u(x, y), sf::Color(pixelValue, pixelValue, pixelValue));
            }
        }

        sf::Texture texture;
        if (!textures[i].loadFromImage(img))
        {
            std::cerr << "Failed to load image into texture at index " << i << std::endl;
        };
        textures.push_back(std::move(texture));

        sf::Sprite sprite = sf::Sprite(texture);
        sprite.setTexture(textures.back());
        sprite.setPosition(sf::Vector2f(static_cast<float>(i * cols), 0));
        sprites.push_back(std::move(sprite));
    };

    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
        };

        window.clear();
        for (const auto &sprite : sprites)
            window.draw(sprite);
        window.display();
    };
};

#endif