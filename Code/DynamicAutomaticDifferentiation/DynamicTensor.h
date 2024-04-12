#pragma once
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "../CommonMathMoudle/Tensor.h"
#include "../CommonDataStructure/HyperElement.h"
#include "../CommonMathMoudle/OpsType.h"

class DynamicTensor;

struct DynamicOps
{
    DynamicOps(){}
    DynamicOps(DynamicTensor* DynamicTensorNode);
    /**算子，叶子节点为Base.*/
    size_t DynamicOpsType;
    /**只有叶子节点有这一项.*/
    DynamicTensor* leafNode = nullptr;
    /**输入节点.*/
    std::vector<DynamicOps>InputOpsList;
};

class DynamicTensor
{
public:

    /**动态张量的成员变量.*/
    std::shared_ptr<Tensor> TensorPointer = nullptr;
    DynamicOps Ops;
    std::vector<DynamicTensor*>OutNodeList;//输出节点，只有在需要求导的时候可用

    /**初始化动态张量.*/
    DynamicTensor(){};
    DynamicTensor(std::shared_ptr<Tensor> InputTensor);
    DynamicTensor(const DynamicTensor& Other);

    /**创建动态张量.*/
    static DynamicTensor CreateDynamicTensor(std::shared_ptr<Tensor> InputTensor);
    static DynamicTensor CreateDynamicTensor(std::vector<size_t>shape, size_t DeviceNum = 0);
    static DynamicTensor CreateDynamicTensor(std::vector<size_t>shape, std::vector<float>InputData, size_t DeviceNum = 0);
    /**创建向量，默认为行向量.*/
    static DynamicTensor CreateVector(std::vector<float>InputData, size_t DeviceNum = 0);

    /**析构函数释放内存.*/
    ~DynamicTensor();

    /**Tensor内函数组装.*/
    void PrintData();

    /**运算符重载.*/

    /**计算图逻辑.*/
    
    /**运算符重载逻辑.*/
    void Set(DynamicTensor* ThisTensor, const DynamicTensor* OtherTensor);

    /**算子.*/
    static DynamicTensor Add(DynamicTensor& InputFirst, DynamicTensor& InputSecond, bool RequiresGrad = false);
};