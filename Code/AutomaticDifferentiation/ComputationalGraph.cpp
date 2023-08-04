
#include "ComputationalGraph.h"
#include "Ops/OpsFactory.h"

ComputationalGraph::ComputationalGraph():BaseGraph()
{

}

ComputationalGraph::~ComputationalGraph()
{
    for(std::map<std::string, BaseNode*>::iterator it = Nodes.begin();it!=Nodes.end();it++)
    {
        delete it->second;
    }
    Nodes.clear();
    Opss.clear();
}

void ComputationalGraph::AddNode(BaseNode* NewNode)
{
    Nodes[NewNode->id] = NewNode;
}

void ComputationalGraph::RegisterNode(std::string id)
{
    AddNode(new ComputationalNode(id));
}

void ComputationalGraph::RegisterVariableNode(std::string Nodeid)
{
    RegisterNode(Nodeid);
    GetNode(Nodeid)->Property.Set("RequireGrad", true);
    GetNode(Nodeid)->Property.Set("Input", true);
}

void ComputationalGraph::RegisterConstNode(std::string Nodeid)
{
    RegisterNode(Nodeid);
    GetNode(Nodeid)->Property.Set("Input", true);
}

ComputationalNode* ComputationalGraph::GetNode(std::string Nodeid)
{
    assert(Nodes.find(std::string(Nodeid))!=Nodes.end() && "This Graph has not this Nodeid ");
    return static_cast <ComputationalNode*>(Nodes[std::string(Nodeid)]);
}

void ComputationalGraph::RegisterOpsAddEdge(std::string OutputNodeid, std::string InputNodeid)
{
    GetNode(OutputNodeid)->InputNodeidList.push_back(InputNodeid);
    GetNode(InputNodeid)->OutputNodeidList.push_back(OutputNodeid);
}

std::string ComputationalGraph::GetDNodeid(std::string id)
{
    return id+"_d";
}

void ComputationalGraph::RegisterOps(std::string OutputNodeid, std::vector<std::string> InputNodeids, size_t OpsTypeid, Dict OpsParams)
{
    Opss[OutputNodeid] = OpsFactory::GetOps(OpsTypeid, OpsParams, this);
    Opss[OutputNodeid]->Nodeid = OutputNodeid;
    for(auto InputNodeid:InputNodeids)
    {
        RegisterOpsAddEdge(OutputNodeid, InputNodeid);
    }
}

void ComputationalGraph::RegisterDNode(std::string Nodeid)
{
    std::string DNodeid = GetDNodeid(Nodeid);
    RegisterNode(DNodeid);
    GetNode(DNodeid)->Property.Set("RequireGrad", true);
    GetNode(Nodeid)->DNodeid = DNodeid;
    RegisterOps(DNodeid, std::vector<std::string>{}, OpsType::Add, Dict());
}

void ComputationalGraph::BackwardGraphBuild()
{
    /**1.首先对每个点建立自己的反向点
     * 2.反向点一定是一个加算子，用来把后续部分的不同求导求和
     * 3.对于后续求导算子，要打求导标记，应该允许让求导标记抹除和普通见图无疑，以供多次求导
     * 4.抹除后也可以提供一个选项，用来标记是几次求导，是谁导出来的
     * */
    std::vector<std::string>NodeidListFromMap;
    for(std::map<std::string, BaseNode*>::iterator it = Nodes.begin();it!=Nodes.end();it++)
    {
        NodeidListFromMap.push_back(it->first);
    }
    for(auto InputNodeidFromMap:NodeidListFromMap)
    {
        if(static_cast<ComputationalNode*>(GetNode(InputNodeidFromMap))->Property.Get<bool>("RequireGrad") == 0)continue;
        RegisterDNode(InputNodeidFromMap);
    }
    for(std::map<std::string, BaseNode*>::iterator it = Nodes.begin();it!=Nodes.end();it++)
    {
        if(static_cast<ComputationalNode*>(it->second)->Property.Get<bool>("Input") == 0)continue;
        if(static_cast<ComputationalNode*>(it->second)->Property.Get<bool>("RequireGrad") == 0)continue;
        if(Opss.find(it->first)==Opss.end())continue;
        Opss[it->first]->Backward();
    }
}

void ComputationalGraph::NodeOpsForward(std::string Nodeid)
{
    if(Opss.find(Nodeid)==Opss.end())return;
    Opss[Nodeid]->Forward();
}

void ComputationalGraph::ForwardDfs(std::string DfsStartNodeid)
{
    if(ComputeFlag.find(DfsStartNodeid) != ComputeFlag.end())return;
    ComputeFlag[DfsStartNodeid] = true;
    ComputationalNode* FoundNode = GetNode(DfsStartNodeid);
    for(int a =0;a<FoundNode->InputNodeidList.size();a++)
    {
        ForwardDfs(FoundNode->InputNodeidList[a]);
    }
    NodeOpsForward(DfsStartNodeid);
}
