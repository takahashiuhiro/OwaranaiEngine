#pragma once
#include "../CommonDataStructure/BaseGraph.h"
#include "ComputationalNode.h"
#include "../CommonDataStructure/Dict.h"
#include "Ops/BaseOps.h"

class ComputationalGraph:public BaseGraph
{
public:

    using OpsMap = std::map<std::string, std::shared_ptr<BaseOps>>;

    OpsMap Opss;
    std::map<std::string, bool>ComputeFlag;
    /**每个算子的反向建图只会执行一次，永不清空.*/
    std::map<std::string, bool>BackwardFlag;
    /**当前哪些节点需要被求导，本轮新增节点不求导.*/
    std::map<std::string, size_t>CurrentBackwardFlag;
    /**当前的求导编号*/
    size_t CurrentBackwardFlagIndex = 0;

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
    /**注册点默认属性.*/
    void RegisterDefaultProperty(std::string Nodeid);
    /**注册变量.*/
    void RegisterVariableNode(std::string Nodeid);
    /**注册权重变量.*/
    void RegisterWeightNode(std::string Nodeid);
    /**注册常量.*/
    void RegisterConstNode(std::string Nodeid);
    /**注册算子.*/
    void RegisterOps(std::string OutputNodeid, std::vector<std::string> InputNodeid, size_t OpsTypeid, Dict OpsParams);
    /**注册算子增边.*/
    void RegisterOpsAddEdge(std::string OutputNodeid, std::string InputNodeid);
    /**一次性注册完算子所有边，并且给默认参数赋值.*/
    void RegisterOpsCompleted(std::string OutputNodeid, std::vector<std::string> InputNodeid, size_t OpsTypeid, Dict OpsParams);
    /**给算子内的输入节点赋予默认参数.*/
    void SetOpsInputNodeDefaultParams(std::string OutputNodeid);
    /**建立反向图.*/
    void BackwardGraphBuild();
    /**注册梯度节点.*/
    void RegisterDNode(std::string id);
    /**对该点返回一个对应的导数节点id，不支持直接注册_d结尾的节点.*/
    std::string GetDNodeid(std::string id);
    /**在算子里会出现a->c且a->b的情况,在这种情况下如果c对a求导，会在a_d和c_d之间的反向过程里建立一个新节点，用来表达c_d给a_d的贡献的中间节点.*/
    std::string GetDPartNodeid(std::string Startid, std::string Endid);
    /**DFS执行得到图中的计算张量.*/
    void ForwardDfs(std::string StartNodeid);
    /**对单个节点算子执行前向.*/
    void NodeOpsForward(std::string DfsStartNodeid);
    /**检查id对应的节点是否存在算子.*/
    bool CheckOps(std::string CheckNodeid);
    /**拿到节点id对应的算子，包一个存在检查.*/
    std::shared_ptr<BaseOps> GetCGOps(std::string OpsNodeid);
    /**清除所有数据，清除后需要把所有叶子节点手动赋值.*/
    void ClearAllData();
    /**是否存在节点.*/
    bool HasNode(std::string InputNode);
    /**是否存在导数节点.*/
    bool HasDNode(std::string InputNode);
    /**将Input属性赋予当前图内所有节点.*/
    void SetAllNodeToInput();
    /**打印计算图的邻接表.*/
    void PrintGraphAdjacencyList(size_t Mode);
    /**需要进行求几次导数的建图.*/
    void BackwardMultiBuildGraph(size_t Times);
    /**以下词条任意为false的将被清理数据.*/
    void ClearDataPropertyExclude(std::vector<std::string>CheckPropertyList);
    /**清除Weight和Const以外的节点.*/
    void ClearDataPropertyExclude();
    /**查询输入节点编号是否是待求导编号.*/
    bool CheckInputNodeidCanBackward(std::string InputNodeid);
};