#pragma once
#include "StdInclude.h"

//把一个向量加到另一个向量上
void AddVectorToVector(float* VectorInput, float* VectorOutput, float Weight, int Length);
//二维矩阵的高斯消元Column >= Row
void MatrixGaussianElimination(float* InputMatrix, int Row, int Column);




