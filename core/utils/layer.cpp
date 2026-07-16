#include "matrix.h"
#include <random>
#include <cmath>
#include "layer.h"


class Layer 
{
    public:
        Matrix Weight;
        Matrix Bias;
        Matrix X;
        float lr;

        Matrix Backward(const Matrix& grad)
        {
            Matrix prev = Multiply_B_T(grad, Weight);

            Matrix dW = Multiply_A_T(X, grad);
            Multiply_Inplace(dW, lr);
            Subtract(Weight, dW);

            Matrix dB = Multiply_Copy(grad, lr); // Replace with SumRows() for batched input
            Subtract(Bias, dB);

            return prev;
        }

        Matrix Forward(const Matrix& X_IN) {
            this->X = X_IN;

            Matrix out = Multiply(X_IN, Weight);

            Add_Bias(out, Bias);

            return out;
        }
};