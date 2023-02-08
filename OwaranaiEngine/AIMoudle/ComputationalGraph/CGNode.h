#pragma once
#include "StdInclude.h"
//#include "../TensorCore/MoudleInclude.h"
#include "../../CommonMathMoudle/MoudleInclude.h"
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
    CGNode(std::string OpsType, bool NeedGradient, Hyperparameter OpsParams);
    /**Init by Input list and Ops*/
    CGNode(std::vector<CGNode*>InputNode, std::string OpsType, bool NeedGradient);
    CGNode(std::vector<CGNode*>InputNode, std::string OpsType, bool NeedGradient, Hyperparameter OpsParams);
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
    /**设置带参数的算子
     * 
    */
    void SetOps(std::string OpsType, Hyperparameter OpsParams);
    /**start computing From Input*/
    void Forward();
    /**build backward node*/
    void BackwardBuild(bool IsOutput);
    /**start Gradient computing from this node to Input*/
    void Backward(Tensor* Loss);
    /**根据输入参数删除节点的content
     * params:
     * NodeTypeList 装标签的lsit
     * IsInclude 如果是True的话那就删除包含NodeTypeList内标签节点的content，False的话删除[不]包含的
    */
    void ClearDataContent(std::vector<std::string>NodeTypeList, bool IsInclude);
    /**递归的删除节点的content
     * params:
     * NodeTypeList 装标签的lsit
     * IsInclude 如果是True的话那就删除包含NodeTypeList内标签节点的content，False的话删除[不]包含的
     * FlagMap 用于记忆化搜索
    */
    void ClearDataDFS(std::vector<std::string>NodeTypeList, bool IsInclude, std::map<CGNode*, bool>*FlagMap);
    /*
     * 该节点为输出节点可用，清理输入节点为inputnodelist的梯度
    */
    void ClearGradient(std::vector<CGNode*>InputNodeList);
    /*
     * 该节点为输出节点可用，清理输入节点为inputnodelist的本体
    */
    void ClearComputeResult(std::vector<CGNode*>InputNodeList);
};
