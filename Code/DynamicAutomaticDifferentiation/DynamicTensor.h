#pragma once
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "../CommonMathMoudle/Tensor.h"
#include "../CommonDataStructure/HyperElement.h"

struct DynamicTensorOpsType
{
    static const int None = 0;
    static const int Add = 1;
};

class DynamicTensor
{
public:

    /**动态张量的成员变量.*/
    Tensor* TensorPointer = nullptr;
    he Params;
    std::vector<DynamicTensor*>InputList;
    std::vector<DynamicTensor*>OutputList;
    std::string id;
    std::shared_ptr<DynamicTensor> Grad = nullptr;//反向节点
    size_t OpsType = 0;
    std::map<DynamicTensor*, he>OpsParams;//存每个节点的output节点的算子信息

    /**初始化动态张量.*/
    DynamicTensor(){};
    DynamicTensor(Tensor* InputTensor);

    static std::shared_ptr<DynamicTensor> CreateDynamicTensor(Tensor* InputTensor);
    static std::shared_ptr<DynamicTensor> CreateDynamicTensor(std::vector<size_t>shape);
    static std::shared_ptr<DynamicTensor> CreateDynamicTensor(std::vector<size_t>shape, size_t DeviceNum);
    static std::shared_ptr<DynamicTensor> CreateDynamicTensor(std::vector<size_t>shape, size_t DeviceNum, std::vector<float>InputData);

    /**初始化.*/
    void DynamicTensorInit();
    void DynamicTensorParamsInit();//初始化张量参数

    /**析构函数释放内存.*/
    ~DynamicTensor();

    /**运算符重载.*/

    /**Tensor内函数组装.*/
    void PrintData();

    /**算子.*/
    static std::shared_ptr<DynamicTensor> Add(DynamicTensor* Input1, DynamicTensor* Input2, bool NeedGrad = false);

};