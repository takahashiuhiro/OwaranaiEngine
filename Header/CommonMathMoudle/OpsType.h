#pragma once
#include <iostream>
#include <string>

namespace OwaranaiEngine
{

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
    static const size_t EleLog;//对元素取log
    static const size_t SubSend;//对元素取log

    static const std::string DumpToString(size_t InputOpsType)
    {
        if (InputOpsType == Base)return "Base";
        if (InputOpsType == NoneOps)return "NoneOps";
        if (InputOpsType == Add)return "Add";
    }
};

const size_t OpsType::Base = 0;//基础
const size_t OpsType::NoneOps = 1;//什么都不做
const size_t OpsType::Add = 2;//矩阵加
const size_t OpsType::EleMul = 3;//元素乘
const size_t OpsType::MatMul = 4;//矩阵乘
const size_t OpsType::BroadCastTo = 5;//矩阵广播
const size_t OpsType::Sum = 6;//矩阵求和
const size_t OpsType::Softmax = 7;
const size_t OpsType::ReLU = 8;
const size_t OpsType::GenerateSign = 9;//生成符号矩阵，没有反向
const size_t OpsType::Pow = 10;//幂次
const size_t OpsType::EleExp = 11;//指数函数
const size_t OpsType::View = 12;//改变张量shape
const size_t OpsType::Transpose = 13;//交换shape
const size_t OpsType::EleLog = 14;//对元素取log
const size_t OpsType::SubSend = 15;//对元素取log

}