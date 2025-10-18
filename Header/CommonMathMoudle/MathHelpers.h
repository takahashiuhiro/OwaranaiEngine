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

namespace OwaranaiEngine
{

//把一个向量加到另一个向量上
void AddVectorToVector(float* VectorInput, float* VectorOutput, float Weight, int Length)
{
    for(int a=0;a<Length;a++)
    {
        VectorOutput[a] += Weight*VectorInput[a];
    }
}
//二维矩阵的高斯消元Column >= Row
void MatrixGaussianElimination(float* InputMatrix, int Row, int Column)
{
    for(int a=0;a<Row;a++)
    {
        //遍历第几个主元
        for(int b=a;b<Row;b++)
        {
            if(!InputMatrix[b*Column+a])continue;
            else
            {
                if(b==a)break;
                for(int c=0;c<Column;c++)
                {
                    float TmpSwap = InputMatrix[b*Column + c];
                    InputMatrix[b*Column + c] = InputMatrix[a*Column + c];
                    InputMatrix[a*Column + c] = TmpSwap;
                }
                break;
            }
        }
        for(int b=0;b<Column;b++)
        {
            if(a==b)continue;
            InputMatrix[a*Column + b] /= InputMatrix[a*Column + a];
        }
        InputMatrix[a*Column+a] = 1.;
        for(int b=0;b<Row;b++)
        {
            if(a == b||!InputMatrix[b*Column+a])continue;
            AddVectorToVector(InputMatrix+a*Column, InputMatrix+b*Column, -1*InputMatrix[b*Column+a], Column);
        }
    }
}
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

template<typename T>
T Factorial(T InputNum)
{
    T Res = 1;
    for(T a=1;a <= InputNum;a=a+1)Res = Res*a;
    return Res;
}

}
