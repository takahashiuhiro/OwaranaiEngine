#pragma once
#include <iostream>

struct OpsType
{
    static const size_t Base = 0;//基础
    static const size_t NoneOps = 1;//什么都不做
    static const size_t Add = 2;//矩阵加
    static const size_t EleMul = 3;//元素乘
    static const size_t MatMul = 4;//矩阵乘
    static const size_t MatMulFirstT = 5;//首矩阵的转置乘
    static const size_t MatMulSecondT = 6;//次矩阵的转置乘
};