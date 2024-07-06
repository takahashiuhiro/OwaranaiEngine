#pragma once
#include <iostream>
#include <string>

struct OpsType
{
    static const size_t Base;//基础
    static const size_t NoneOps;//什么都不做
    static const size_t Add ;//矩阵加
    static const size_t EleMul ;//元素乘
    static const size_t MatMul ;//矩阵乘
    static const size_t BroadCastTo ;//矩阵广播
    static const size_t Sum ;//矩阵求和
    static const size_t Softmax ;
    static const size_t ReLU ;
    static const size_t GenerateSign ;//生成符号矩阵，没有反向
    static const size_t Pow ;//幂次
    static const size_t EleExp ;//指数函数
    static const size_t View;//改变张量shape
    static const size_t Transpose;//交换shape

    static const std::string DumpToString(size_t InputOpsType)
    {
        if (InputOpsType == Base)return "Base";
        if (InputOpsType == NoneOps)return "NoneOps";
        if (InputOpsType == Add)return "Add";
    }
};