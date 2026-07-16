#pragma once

#include <vector>

class Matrix
{
public:
    int rows;
    int cols;
    std::vector<float> data;

    Matrix();
    Matrix(int rows, int cols);
};

Matrix Multiply(const Matrix& A, const Matrix& B);
Matrix Multiply_A_T(const Matrix& A, const Matrix& B);
Matrix Multiply_B_T(const Matrix& A, const Matrix& B);
void Multiply(Matrix& A, float scalar);
void Subtract(Matrix& A, const Matrix& B);
Matrix Multiply_Copy(const Matrix& A, float scalar);
void Add(Matrix& A, const Matrix& B);
void Add_Bias(Matrix& A, const Matrix& bias);
void Multiply_Inplace(Matrix& A, float scalar);