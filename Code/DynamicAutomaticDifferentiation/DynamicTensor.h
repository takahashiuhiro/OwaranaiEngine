#pragma once
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "../CommonMathMoudle/Tensor.h"
#include "../CommonDataStructure/HyperElement.h"

class DynamicTensor
{
public:

    /**动态张量的成员变量.*/
    Tensor* TensorPointer = nullptr;
    std::string id;

    /**初始化动态张量.*/
    DynamicTensor(){};
    DynamicTensor(Tensor* InputTensor);

    /**创建动态张量.*/
    static DynamicTensor CreateDynamicTensor(Tensor* InputTensor);
    static DynamicTensor CreateDynamicTensor(std::vector<size_t>shape);
    static DynamicTensor CreateDynamicTensor(std::vector<size_t>shape, size_t DeviceNum);
    static DynamicTensor CreateDynamicTensor(std::vector<size_t>shape, size_t DeviceNum, std::vector<float>InputData);
    /**创建向量，默认为行向量.*/
    static DynamicTensor CreateVector(std::vector<float>InputData, size_t DeviceNum = 0);

    /**析构函数释放内存.*/
    ~DynamicTensor();

    /**运算符重载.*/
    DynamicTensor operator + (DynamicTensor& Other);
    DynamicTensor operator + (DynamicTensor&& Other);

    /**运算符重载执行逻辑.*/
    Tensor* OperatorPlus(Tensor*OtherDynamicTensor);

    /**Tensor内函数组装.*/
    void PrintData();
    void Backward(DynamicTensor* Input = nullptr);

    /**算子.*/


};