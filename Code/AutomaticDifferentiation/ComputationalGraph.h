#pragma once
#include <cassert>
#include "../CommonDataStructure/BaseGraph.h"
#include "ComputationalNode.h"
#include "../CommonDataStructure/Dict.h"
#include "Ops/BaseOps.h"

class ComputationalGraph:public BaseGraph
{
public:

    using OpsMap = std::map<std::string, std::shared_ptr<BaseOps>>;

    OpsMap Opss;

    ComputationalGraph();
    ~ComputationalGraph();

    virtual void AddNode(BaseNode* NewNode);
    virtual void AddEdge(BaseEdge* NewEdge){};

    /**返回一个Node.*/
    ComputationalNode* GetNode(std::string Nodeid);

    /**初始化.*/
    void CommonInit();

    /**注册点.*/
    void RegisterNode(std::string id);
    /**注册算子.*/
    void RegisterOps(std::string OutputNodeid, std::vector<std::string> InputNodeid, size_t OpsTypeid, Dict OpsParams);
    /**注册算子增边.*/
    void RegisterOpsAddEdge(std::string OutputNodeid, std::string InputNodeid);

    /**建立反向图.*/
    void BackwardGraphBuild();
    /**注册梯度节点.*/
    void RegisterDNode(std::string id);

    /**对该点返回一个对应的导数节点id，不支持直接注册_d结尾的节点.*/
    std::string GetDNodeid(std::string id){return id+"_d";}


};