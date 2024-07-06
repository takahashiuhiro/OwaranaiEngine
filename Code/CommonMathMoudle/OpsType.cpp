#include "OpsType.h"

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