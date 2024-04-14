#pragma once
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <set>
#include "../CommonDataStructure/HyperElement.h"
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
    he Params;
};

class DynamicTensor
{
public:

    /**动态张量的成员变量.*/
    std::shared_ptr<Tensor> TensorPointer = nullptr;
    DynamicOps Ops;
    std::set<DynamicTensor*>OutNodeList;//输出节点，只有在需要求导的时候可用
    bool RequiresGrad = 0;
    std::shared_ptr<DynamicTensor>Grad = nullptr;

    /**初始化动态张量.*/
    DynamicTensor(){};
    DynamicTensor(std::shared_ptr<Tensor> InputTensor, bool IsRequiresGrad = false);
    DynamicTensor(const DynamicTensor& Other);

    /**创建动态张量.*/
    static DynamicTensor CreateDynamicTensor(std::shared_ptr<Tensor> InputTensor, bool IsRequiresGrad = false);
    static DynamicTensor CreateDynamicTensor(std::vector<size_t>shape, bool IsRequiresGrad = false, size_t DeviceNum = 0);
    static DynamicTensor CreateDynamicTensor(std::vector<size_t>shape, std::vector<float>InputData, bool IsRequiresGrad = false, size_t DeviceNum = 0);
    /**创建向量，默认为行向量.*/
    static DynamicTensor CreateVector(std::vector<float>InputData, size_t DeviceNum = 0);

    /**析构函数释放内存.*/
    ~DynamicTensor();
    void SetOutputList(DynamicOps&CurOps,DynamicTensor* TargetOutputNode);
    void SetInputList(DynamicOps& CurOps, DynamicTensor* TargetOutputNode);

    /**Tensor内函数组装.*/
    void PrintData();

    /**运算符重载.*/

    /**计算图逻辑.*/
    static void SetForwardHistory(DynamicTensor&InputRes ,size_t InputOptType, std::vector<DynamicTensor*>OpsList,he InputPrams, bool IsRequiresGrad);
    void Backward(DynamicTensor LossResult);//输入一个反向向量开始反传
    void Backward();//输入一个反向向量开始反传,这是起点位置
    void BackwardDfs(std::map<DynamicOps*, std::shared_ptr<DynamicTensor>>& GradGlobalResult, std::map<DynamicOps*, std::shared_ptr<DynamicTensor>>&ForwardGlobalResult, std::map<std::pair<DynamicOps*, DynamicOps*>, std::shared_ptr<DynamicTensor>>& PartGradGlobalResult, DynamicOps* CurOps);
    
    /**运算符重载逻辑.*/
    void Set(DynamicTensor* ThisTensor, const DynamicTensor* OtherTensor);

    /**算子.*/
    static DynamicTensor Add(std::vector<DynamicTensor*>InputList, he InputPrams, bool RequiresGrad = false);
};