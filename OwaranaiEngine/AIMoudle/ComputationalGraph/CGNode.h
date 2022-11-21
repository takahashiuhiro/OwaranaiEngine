#pragma once
#include "StdInclude.h"
#include "../TensorCore/MoudleInclude.h"
#include "../Ops/MoudleInclude.h"

struct CGNode
{
public:
    CGNode(){}
    /**Init by NeedGradient*/
    CGNode(bool NeedGradient);
    /**Init by tensor*/
    CGNode(Tensor* NodeContent, bool NeedGradient);
    /**Init by Ops*/
    CGNode(std::string OpsType, bool NeedGradient);
    /**Init by Input list and Ops*/
    CGNode(std::vector<CGNode*>InputNode, std::string OpsType, bool NeedGradient);
    /**result of the input node computing by ops */
    Tensor* NodeContent= nullptr;
    /**require backward*/
    bool NeedGradient = 0;
    /**Input list*/
    std::vector<CGNode*>InputNode;
    /**node Derivative*/
    CGNode* DerivativeNode = nullptr;
    /**Ops*/
    BaseOps<CGNode,Tensor>* FunOps;
    /**Ops Type*/
    std::string OpsType;
    /**
     * if we need to make a node difference with other nodes or make a set of node have a difference process, insert a tuple to nodetype
     * Params
    */
    std::map<std::string, bool>NodeType;
    /**backward build flag, if true return dfs*/
    bool BackwardBuildFlag = 0;
    /**Get ops pointer*/
    void SetOps(std::string OpsType);
    /**start computing From Input*/
    void Forward();
    /**build backward node*/
    void BackwardBuild(bool IsOutput);
    /**start Gradient computing from this node to Input*/
    void Backward(Tensor* Loss);
    /**clear tensor content by nodetype*/
    void ClearDataContent(std::vector<std::string>NodeTypeList);
};
