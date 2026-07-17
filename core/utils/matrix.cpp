#include <openblas/cblas.h>
#include <iterator>
#include "matrix.h"


Matrix::Matrix()
    : rows(0), cols(0)
{
}

Matrix::Matrix(int r, int c)
    : rows(r), cols(c), data(r * c, 0.0f)
{
}

Matrix Matrix::row(int i) const
{
    Matrix r(1, cols);
    const float* src = data.data() + i * cols;
    for (int c = 0; c < cols; ++c)
        r.data[c] = src[c];
    return r;
}

Matrix Multiply(const Matrix& A, const Matrix& B) 
{

    Matrix C(A.rows, B.cols);

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        A.rows,
        B.cols,
        A.cols,
        1.0f,
        A.data.data(),
        A.cols,
        B.data.data(),
        B.cols,
        0.0f,
        C.data.data(),
        B.cols
    );

    return C;
}


Matrix Multiply_A_T(const Matrix& A, const Matrix& B) 
{
    // Computes A^T * B
    // A is (m x k), A^T is (k x m), B is (m x n) => C is (k x n)
    Matrix C(A.cols, B.cols);

    cblas_sgemm(
        CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        A.cols,   // rows of A^T
        B.cols,   // cols of B
        A.rows,   // inner dimension
        1.0f,
        A.data.data(),
        A.cols,
        B.data.data(),
        B.cols,
        0.0f,
        C.data.data(),
        B.cols
    );

    return C;
}

Matrix Multiply_B_T(const Matrix& A, const Matrix& B) 
{
    // Computes A * B^T
    // A is (m x k), B^T is (k x n) => B is (n x k) => C is (m x n)
    Matrix C(A.rows, B.rows);

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        A.rows,   // rows of A
        B.rows,   // rows of B (cols of B^T)
        A.cols,   // inner dimension
        1.0f,
        A.data.data(),
        A.cols,
        B.data.data(),
        B.cols,
        0.0f,
        C.data.data(),
        B.rows
    );

    return C;
}

void Multiply(Matrix& A, float scalar)
{
    cblas_sscal(
        A.data.size(),
        scalar,
        A.data.data(),
        1
    );
}

Matrix Multiply_Copy(const Matrix& A, float scalar)
{
    Matrix C = A; 

    cblas_sscal(
        C.data.size(),
        scalar,
        C.data.data(),
        1
    );

    return C;
}

void Subtract(Matrix& A, const Matrix& B)
{
    cblas_saxpy(
        A.data.size(),      // Number of elements
        -1.0f,              // Alpha
        B.data.data(),      // X
        1,
        A.data.data(),      // Y
        1
    );
}

void Add(Matrix& A, const Matrix& B)
{
    cblas_saxpy(
        A.data.size(),
        1.0f,
        B.data.data(),
        1,
        A.data.data(),
        1
    );
}

void Add_Bias(Matrix& A, const Matrix& bias)
{
    for (int i = 0; i < A.rows; i++)
    {
        cblas_saxpy(
            A.cols,
            1.0f,
            bias.data.data(),
            1,
            A.data.data() + i * A.cols,
            1
        );
    }
}

void Multiply_Inplace(Matrix& A, float scalar)
{
    cblas_sscal(
        static_cast<int>(A.data.size()), // Number of elements
        scalar,                          // Scale factor
        A.data.data(),                   // Matrix data
        1                                // Stride
    );
}