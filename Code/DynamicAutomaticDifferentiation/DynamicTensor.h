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
    /**算子成员变量.*/
    he Params;//算子参数
    size_t DynamicOpsType;//算子类型，叶子节点为Base
    DynamicTensor* LeafNode = nullptr;//指向算子的结算张量，如果为nullptr代表张量被删除
    std::vector<std::shared_ptr<DynamicOps>>InputOpsList;//输入节点，需要后继保存输入的资源
    std::set<DynamicOps*>OutputOpsSet;//输出节点，不需要保存output的资源
    std::shared_ptr<DynamicOps>GradOps = nullptr;
    std::shared_ptr<Tensor> TensorPointer = nullptr;//存的张量内容
    bool RequiresGrad = false;
};

class DynamicTensor
{
public:

    /**动态张量的成员变量.*/
    std::shared_ptr<DynamicOps>Ops = nullptr;//每个动态张量的算子，如果张量被删掉但是需要计算图，这个算子可以交出去，交出去的时候需要删掉算子中的leafNode变量为nullptr
    std::shared_ptr<DynamicTensor>Grad = nullptr;
    std::map<size_t, DynamicTensor(*)(std::vector<DynamicTensor>, he, bool)>ForwardOpsMap;
    //std::map<size_t, DynamicTensor(*)(std::vector<DynamicTensor*>, he, bool)>BackwardOpsMap;应该有，但可能不是这个类型

    /**内存管理.*/
    DynamicTensor();//初始化动态张量
    DynamicTensor(std::shared_ptr<Tensor> InputTensorPointer, bool InputRequiresGrad = 0);
    DynamicTensor(std::shared_ptr<DynamicOps>InputOps);
    void OpsSetInMap();

    ~DynamicTensor();//析构函数释放内存

    /**Tensor内函数组装.*/

    /**运算符重载.*/

    /**计算图逻辑.*/
    static DynamicTensor SetComputationalHistory(Tensor* ResTensor, std::vector<DynamicTensor>InputList, he InputPrams,size_t InputOpsType, bool RequiresGrad);
    void Backward();//从这里开始反向传播,

    /**运算符重载逻辑.*/

    /**算子.*/
    static DynamicTensor Add(std::vector<DynamicTensor>InputList, he InputPrams, bool RequiresGrad = false);
};