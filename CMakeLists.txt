cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)

project(gan)

# Find Torch
set(Torch_DIR "/Users/anshumantiwari/Documents/codes/personal/C++/GAN/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Find OpenCV
set(OpenCV_DIR "/opt/homebrew/Cellar/opencv/4.11.0_1/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Find SFML 3 with explicit components
set(SFML_DIR "/opt/homebrew/Cellar/sfml/3.0.0/lib/cmake/SFML")
find_package(SFML 3 COMPONENTS Graphics Window System REQUIRED)
include_directories(/opt/homebrew/Cellar/sfml/3.0.0/include)
link_directories(/opt/homebrew/Cellar/sfml/3.0.0/lib)

# Explicitly set include directories for Torch
include_directories(
    /Users/anshumantiwari/Documents/codes/personal/C++/GAN/libtorch/include
    /Users/anshumantiwari/Documents/codes/personal/C++/GAN/libtorch/include/torch/csrc/api/include
)
link_directories(/Users/anshumantiwari/Documents/codes/personal/C++/GAN/libtorch/lib)

# Compiler Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
string(REPLACE "--no-as-needed" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "--as-needed" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Define executable
add_executable(gan gan.cpp)

# Link against Torch, OpenCV, and SFML
target_link_libraries(gan 
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS} 
    SFML::Graphics 
    SFML::Window 
    SFML::System
)

# Set C++ Standard
set_property(TARGET gan PROPERTY CXX_STANDARD 17)