#pragma once
#include "StdInclude.h"

//把一个向量加到另一个向量上
void AddVectorToVector(float* VectorInput, float* VectorOutput, float Weight, int Length);
//二维矩阵的高斯消元Column >= Row
void MatrixGaussianElimination(float* InputMatrix, int Row, int Column);
//标量乘法快速幂
template<typename T>
T BinaryExp(T Base, int Num)
{
    T Res = T(1.);
    T NewBase = Base;
    while(Num)
    {
        if(Num&1)
        {
            Res*=NewBase;
        }
        NewBase*=NewBase;
        Num >>= 1;
    }
    return Res;
}



