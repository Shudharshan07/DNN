#pragma once

#include "../matrix.h"
#include <vector>

class Data
{
public:
    Matrix X;
    Matrix Y;

    // Default constructor: generates 100 samples of  y = 2x + 3
    // X in [-5, 5] evenly spaced, then shuffled.
    Data();

    Data(const std::vector<float>& x,
         const std::vector<float>& y,
         int x_cols = 1,
         int y_cols = 1);

    int size() const { return X.rows; }

    

private:
    void shuffle_();
};
