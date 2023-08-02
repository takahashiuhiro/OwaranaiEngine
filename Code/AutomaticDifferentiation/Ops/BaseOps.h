#pragma once
#include "../../CommonMathMoudle/Tensor.h"
#include "../../CommonDataStructure/Dict.h"


class ComputationalGraph;

//template<typename ComputationalGraph>
class BaseOps
{
public:
    size_t OpsTypeName;
    Dict Params;
    ComputationalGraph* CG;
    std::string Nodeid;

    virtual ~BaseOps(){};
    
    /**前向计算.*/
    virtual void Forward() = 0;
    /**反向计算图设置.*/
    virtual void Backward() = 0;
    /**初始设置参数权重.*/
    virtual void ParamsDefinition() = 0;

    void CommonInit(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG);
};
