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
    std::map<size_t, void(*)(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& ,std::shared_ptr<DynamicOps>)>BackwardOps;//反向函数map

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
    void Backward(DynamicTensor* Loss = nullptr);//从这里开始反向传播
    void BackwardDFS(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>&BackwardOpsMap, std::map<DynamicOps*, int>& OutputSetSize, DynamicTensor* Loss, std::shared_ptr<DynamicOps>CurOps);
    bool CheckPartialGradReady(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>& BackwardOpsMap, std::map<DynamicOps*, int>& OutputSetSize, std::shared_ptr<DynamicOps>CurOps);
    void GenEmptyGradDynamicTensor(DynamicTensor* Loss);
    void GetAllOutputSizeBeforeBackward(std::map<DynamicOps*, int>& OutputSetSize,std::shared_ptr<DynamicOps>CurOps);

    /**运算符重载逻辑.*/

    /**算子.*/
    static DynamicTensor DynamicStdOps_Forward_Add(std::vector<DynamicTensor>InputList, he InputPrams, bool RequiresGrad = false);
    static void DynamicStdOps_Backward_Add(std::map<DynamicOps*, std::map<DynamicOps*, std::shared_ptr<DynamicOps>>>&BackwardOpsMap,std::shared_ptr<DynamicOps>CurOps);
};