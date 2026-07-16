#pragma once

#include "../utils/matrix.h"

class MSE
{
public:
    // Compute mean squared error
    float operator()(const Matrix& Y_actual, const Matrix& Y_pred) const;

    // Compute dLoss/dPrediction
    Matrix Derivative(const Matrix& Y_actual, const Matrix& Y_pred) const;
};