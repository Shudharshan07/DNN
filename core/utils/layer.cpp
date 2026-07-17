#include "matrix.h"
#include <random>
#include <cmath>
#include "layer.h"

Matrix Layer::Forward(const Matrix& X_IN)
{
    X = X_IN;

    Matrix out = Multiply(X_IN, Weight);
    Add_Bias(out, Bias);

    return out;
}

Matrix Layer::Backward(const Matrix& grad)
{
    Matrix prev = Multiply_B_T(grad, Weight);

    Matrix dW = Multiply_A_T(X, grad);
    Multiply_Inplace(dW, lr);
    Subtract(Weight, dW);

    Matrix dB = Multiply_Copy(grad, lr);
    Subtract(Bias, dB);

    return prev;
}