#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <map>
#include <any>
#include <cmath>
#include <thread>
#include <iomanip>

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

template<typename T, typename Y>
std::vector<T> MathArange(T Start, T End, Y Step)
{
    std::vector<T>Res;
    for(T It = Start;It < End;It = It + Step)
    {
        Res.push_back(It);
    }
    return Res;
}

