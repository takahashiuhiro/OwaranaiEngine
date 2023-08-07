#pragma once
#include "../../CommonMathMoudle/Tensor.h"
#include "../../CommonDataStructure/Dict.h"
#include "OpsType.h"

class ComputationalGraph;
class BaseOps
{
public:
    size_t OpsTypeName;
    Dict Params;
    ComputationalGraph* CG;
    std::string Nodeid;

    using AddWeightType = std::map<std::string, float>;
    using AddWeightTypePtr = std::shared_ptr<AddWeightType>;

    virtual ~BaseOps(){};
    /**前向计算.*/
    virtual void Forward() = 0;
    /**反向计算图设置.*/
    virtual void Backward() = 0;
    /**初始设置参数权重.*/
    virtual void ParamsDefinition();
    /**算子初始化.*/
    void CommonInit(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
    /**设置每个输入参数的常系数.*/
    void SetAddWeight(std::map<std::string, float> InputNodeWeight);
    /**获取输入参数的常系数.*/
    float GetAddWeight(std::string InputNodeid);
    /**获取节点的输入节点列表.*/
    std::vector<std::string> &GetInputNodeList();
    std::vector<std::string> &GetInputNodeList(std::string InputNodeid);
};
