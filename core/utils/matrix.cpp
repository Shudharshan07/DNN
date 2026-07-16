#include <openblas/cblas.h>
#include <iterator>
#include "matrix.h"


Matrix::Matrix()
    : rows(0), cols(0)
{
}

Matrix::Matrix(int r, int c)
    : rows(r), cols(c), data(r * c, 1)
{
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

    Matrix C(A.rows, B.cols);

    cblas_sgemm(
        CblasRowMajor,
        CblasTrans,
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

Matrix Multiply_B_T(const Matrix& A, const Matrix& B) 
{

    Matrix C(A.rows, B.cols);

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
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

void Multiply(Matrix& A, float scalar)
{
    cblas_sscal(
        A.data.size(),
        scalar,
        A.data.data(),
        1
    );
}

Matrix Multiply_Copy(Matrix& A, float scalar)
{
    Matrix C = A; 

    cblas_sscal(
        A.data.size(),
        scalar,
        A.data.data(),
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