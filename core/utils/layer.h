#pragma once

#include "matrix.h"
#include <random>
#include <cmath>

inline void He_Init(Matrix& W)
{
    float stddev = std::sqrt(2.0f / static_cast<float>(W.rows));

    static std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, stddev);

    for (float& x : W.data)
        x = dist(rng);
}

class Layer
{
public:
    Matrix Weight;
    Matrix Bias;
    Matrix X;
    float lr;

    Layer(int IN, int OUT)
    {
        Weight = Matrix(IN, OUT);
        He_Init(Weight);

        Bias = Matrix(1, OUT);

        lr = 0.01f;
    }

    Layer(int IN, int OUT, float learning_rate)
    {
        Weight = Matrix(IN, OUT);
        He_Init(Weight);

        Bias = Matrix(1, OUT);

        lr = learning_rate;
    }

    Matrix Forward(const Matrix& X_IN);
    Matrix Backward(const Matrix& grad);
};