#pragma once
#include <iostream>

struct OpsType
{
    static const size_t Base = 0;//基础
    static const size_t NoneOps = 1;//什么都不做
    static const size_t Add = 2;//矩阵加
    static const size_t EleMul = 3;//元素乘
    static const size_t MatMul = 4;//矩阵乘
    static const size_t BroadCastTo = 5;//矩阵广播
    static const size_t Sum = 6;//矩阵求和
    static const size_t Softmax = 7;
    static const size_t ReLU = 8;
    static const size_t GenerateSign = 9;//生成符号矩阵，没有反向
    static const size_t Pow = 10;//生成符号矩阵，没有反向
};