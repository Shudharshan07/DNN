#pragma once

#include "../matrix.h"

class MSE
{
public:
    float operator()(const Matrix& Y_actual, const Matrix& Y_pred) const;

    float Loss(const Matrix& Y_actual, const Matrix& Y_pred) const
    {
        return (*this)(Y_actual, Y_pred);
    }

    Matrix Derivative(const Matrix& Y_actual, const Matrix& Y_pred) const;
};