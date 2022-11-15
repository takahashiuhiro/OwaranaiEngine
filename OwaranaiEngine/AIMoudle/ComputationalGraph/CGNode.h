#pragma once
#include "StdInclude.h"
#include "../TensorCore/MoudleInclude.h"
#include "../Ops/MoudleInclude.h"

struct CGNode
{
public:
    CGNode(){}
    /**Init by tensor*/
    CGNode(Tensor* NodeContent, bool NeedGradient);
    CGNode(bool NeedGradient);
    /**Init by input node, Only use Input Node*/
    CGNode(std::vector<CGNode*>InputNode, std::string OpsType, bool NeedGradient);
    /**result of the input node computing by ops */
    Tensor* NodeContent= nullptr;
    bool NeedGradient = 0;
    /**Input list*/
    std::vector<CGNode*>InputNode;
    /**node Derivative*/
    CGNode* DerivativeNode = nullptr;
    /**Ops*/
    BaseOps<CGNode,Tensor>* FunOps;
    /**Ops Type*/
    std::string OpsType;
    /**Get ops pointer*/
    void SetOps(std::string OpsType);
    /**start computing From Input*/
    void Forward();
    /**start Gradient computing from this node to Input*/
    void Backward(std::string BackType, Tensor* Loss);
};
