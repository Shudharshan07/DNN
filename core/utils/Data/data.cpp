#include "data.h"

#include <algorithm>    // std::iota, std::shuffle
#include <numeric>      // std::iota
#include <random>       // std::mt19937, std::random_device
#include <stdexcept>    // std::invalid_argument


static std::vector<int> random_permutation(int n)
{
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    static std::mt19937 rng(std::random_device{}());
    std::shuffle(perm.begin(), perm.end(), rng);

    return perm;
}

static void copy_row(const Matrix& src, int src_row,
                     Matrix&       dst, int dst_row)
{
    int cols = src.cols;
    const float* s = src.data.data() + src_row * cols;
    float*       d = dst.data.data() + dst_row * cols;

    for (int c = 0; c < cols; ++c)
        d[c] = s[c];
}

Data::Data()
{
    const int N = 100;

    X = Matrix(N, 1);
    Y = Matrix(N, 1);

    for (int i = 0; i < N; ++i)
    {
        float x_val = -5.0f + (10.0f / (N - 1)) * i;   // evenly spaced in [-5, 5]
        float y_val = 2.0f * x_val + 3.0f;              // y = 2x + 3

        X.data[i] = x_val;
        Y.data[i] = y_val;
    }

    shuffle_();
}

Data::Data(const std::vector<float>& x,
           const std::vector<float>& y,
           int x_cols,
           int y_cols)
{
    if (x_cols < 1 || y_cols < 1)
        throw std::invalid_argument("x_cols and y_cols must be >= 1");

    int n_x = static_cast<int>(x.size()) / x_cols;
    int n_y = static_cast<int>(y.size()) / y_cols;

    if (n_x != n_y)
        throw std::invalid_argument("X and Y must have the same number of samples");

    int N = n_x;

    X = Matrix(N, x_cols);
    Y = Matrix(N, y_cols);

    X.data = x;   
    Y.data = y;  

    shuffle_();
}

void Data::shuffle_()
{
    int N = X.rows;
    std::vector<int> perm = random_permutation(N);

    Matrix X_new(N, X.cols);
    Matrix Y_new(N, Y.cols);

    for (int i = 0; i < N; ++i)
    {
        copy_row(X, perm[i], X_new, i);
        copy_row(Y, perm[i], Y_new, i);
    }

    X = std::move(X_new);
    Y = std::move(Y_new);
}

