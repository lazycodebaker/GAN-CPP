#include "matplotlib-cpp/matplotlibcpp.h"
#include <vector>
#include <cmath>

namespace plt = matplotlibcpp;

int main()
{
    std::vector<std::vector<double>> x, y, z;
    for (double i = -5; i <= 5;  i += 0.25) {
        std::vector<double> x_row, y_row, z_row;
        for (double j = -5; j <= 5; j += 0.25) {
            x_row.push_back(i);
            y_row.push_back(j);
            z_row.push_back(::std::sin(::std::hypot(i, j)));
        }
        x.push_back(x_row);
        y.push_back(y_row);
        z.push_back(z_row);
    }

    plt::figure(); 
    plt::plot_surface(x, y, z);
    plt::show();
}



/*
g++ -std=c++17 \
    -I$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
    -I$(python3 -c "import numpy; print(numpy.get_include())") \
    -L/Users/anshumantiwari/miniconda3/lib/ -lpython3.12 \
    ./main.cpp
    
    export DYLD_LIBRARY_PATH=/Users/anshumantiwari/miniconda3/lib:$DYLD_LIBRARY_PATH


*/