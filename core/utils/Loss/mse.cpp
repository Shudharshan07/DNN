#include "mse.h"

#include <openblas/cblas.h>
#include <stdexcept>

float MSE::operator()(const Matrix& Y_actual,
                      const Matrix& Y_pred) const
{
    if (Y_actual.rows != Y_pred.rows ||
        Y_actual.cols != Y_pred.cols)
    {
        throw std::runtime_error("MSE: Matrix dimensions do not match.");
    }

    const int N = static_cast<int>(Y_actual.data.size());

    // diff = Y_pred
    Matrix diff = Y_pred;

    // diff = diff - Y_actual
    cblas_saxpy(
        N,
        -1.0f,
        Y_actual.data.data(),
        1,
        diff.data.data(),
        1);

    // sum(diff^2)
    float squared_sum = cblas_sdot(
        N,
        diff.data.data(),
        1,
        diff.data.data(),
        1);

    return squared_sum / static_cast<float>(N);
}

Matrix MSE::Derivative(const Matrix& Y_actual,
                       const Matrix& Y_pred) const
{
    if (Y_actual.rows != Y_pred.rows ||
        Y_actual.cols != Y_pred.cols)
    {
        throw std::runtime_error("MSE: Matrix dimensions do not match.");
    }

    const int N = static_cast<int>(Y_actual.data.size());

    // grad = Y_pred
    Matrix grad = Y_pred;

    // grad = grad - Y_actual
    cblas_saxpy(
        N,
        -1.0f,
        Y_actual.data.data(),
        1,
        grad.data.data(),
        1);

    // grad *= 2/N
    const float scale = 2.0f / static_cast<float>(N);

    cblas_sscal(
        N,
        scale,
        grad.data.data(),
        1);

    return grad;
}