#include "../CommonDataStructure/Log.h"
#include "ComputationalGraph.h"
#include "Ops/OpsFactory.h"
#include "../CommonDataStructure/CommonFuncHelpers.h"

ComputationalGraph::ComputationalGraph():BaseGraph(){}

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

void ComputationalGraph::RegisterDefaultProperty(std::string Nodeid)
{
    GetNode(Nodeid)->Property.Set("RequireGrad", false);
    GetNode(Nodeid)->Property.Set("Input", false);
    GetNode(Nodeid)->Property.Set("Const", false);
    GetNode(Nodeid)->Property.Set("Weight", false);
}

void ComputationalGraph::RegisterNode(std::string id)
{
    AddNode(new ComputationalNode(id));
    RegisterDefaultProperty(id);
}

void ComputationalGraph::RegisterNode(std::string id, std::vector<size_t>ThisNodeShape)
{
    RegisterNode(id);
    GetNode(id)->NodeContentShape = ThisNodeShape;
}

void ComputationalGraph::RegisterVariableNode(std::string Nodeid)
{
    RegisterNode(Nodeid);
    GetNode(Nodeid)->Property.Set("RequireGrad", true);
    GetNode(Nodeid)->Property.Set("Input", true);
}

void ComputationalGraph::RegisterVariableNode(std::string Nodeid, std::vector<size_t>ThisNodeShape)
{
    RegisterVariableNode(Nodeid);
    GetNode(Nodeid)->NodeContentShape = ThisNodeShape;
}

void ComputationalGraph::RegisterWeightNode(std::string Nodeid)
{
    RegisterVariableNode(Nodeid);
    GetNode(Nodeid)->Property.Set("Weight", true);
}

void ComputationalGraph::RegisterWeightNode(std::string Nodeid, std::vector<size_t>ThisNodeShape)
{
    RegisterWeightNode(Nodeid);
    GetNode(Nodeid)->NodeContentShape = ThisNodeShape;
}

void ComputationalGraph::RegisterConstNode(std::string Nodeid)
{
    RegisterNode(Nodeid);
    GetNode(Nodeid)->Property.Set("Input", true);
    GetNode(Nodeid)->Property.Set("Const", true);
}

void ComputationalGraph::RegisterConstNode(std::string Nodeid, std::vector<size_t>ThisNodeShape)
{
    RegisterConstNode(Nodeid);
    GetNode(Nodeid)->NodeContentShape = ThisNodeShape;
}

ComputationalNode* ComputationalGraph::GetNode(std::string Nodeid)
{
    Log::Assert(HasNode(Nodeid), std::string("This Graph has not this Nodeid :") + Nodeid);
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

std::string ComputationalGraph::GetDPartNodeid(std::string Startid, std::string Endid)
{
    /**Start是输入图的前序，输出的名称是导数图的后继.*/
    return std::string("(")+ GetDNodeid(Endid) + std::string("->") + GetDNodeid(Startid) + std::string(")");
}

std::string ComputationalGraph::GetNodeidByOps(size_t OpsName, std::vector<std::string>InputNodeNameArray)
{
    std::string PreStr = std::string("({Ops:") + NumberToString(OpsName) +std::string("},{NodeName:");
    for(size_t a = 0;a < InputNodeNameArray.size();a++)
    {
        PreStr += InputNodeNameArray[a] + std::string(",");
    }
    PreStr += std::string("})");
    return PreStr;
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

void ComputationalGraph::SetOpsInputNodeDefaultParams(std::string OutputNodeid)
{
    auto ThisOps = GetCGOps(OutputNodeid);
    auto NodeidList = ThisOps->GetInputNodeList();
    /**默认变量系数为1.*/
    for(size_t a = 0;a<NodeidList.size();a++)
    {
        ThisOps->SetAddWeight({{NodeidList[a],1.}});
    }
    /**默认输入矩阵为不转置矩阵.*/
    for(size_t a = 0;a<NodeidList.size();a++)
    {
        ThisOps->SetT({{NodeidList[a],false}});
    }
}

void ComputationalGraph::RegisterOpsCompleted(std::string OutputNodeid, std::vector<std::string> InputNodeid, size_t OpsTypeid, Dict OpsParams)
{
    RegisterOps(OutputNodeid, InputNodeid, OpsTypeid, OpsParams);
    SetOpsInputNodeDefaultParams(OutputNodeid);
}

void ComputationalGraph::RegisterDNode(std::string Nodeid)
{
    std::string DNodeid = GetDNodeid(Nodeid);
    RegisterNode(DNodeid);
    GetNode(DNodeid)->Property.Set("RequireGrad", true);
    GetNode(Nodeid)->DNodeid = DNodeid;
    GetNode(DNodeid)->NodeContentShape = GetNode(Nodeid)->NodeContentShape;
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
        if(HasDNode(InputNodeidFromMap))continue;
        RegisterDNode(InputNodeidFromMap);
    }

    for(auto InputNodeidFromMap:NodeidListFromMap)
    {
        if(GetNode(InputNodeidFromMap)->Property.Get<bool>("Input") == 0)continue;
        if(GetNode(InputNodeidFromMap)->Property.Get<bool>("RequireGrad") == 0)continue;
        if(!CheckOps(InputNodeidFromMap))continue;
        if(BackwardFlag.find(InputNodeidFromMap)!=BackwardFlag.end())continue;
        GetCGOps(InputNodeidFromMap)->Backward();
        BackwardFlag[InputNodeidFromMap] = true;
    }

}

bool ComputationalGraph::CheckOps(std::string CheckNodeid)
{
    return Opss.find(CheckNodeid)!=Opss.end();
}

void ComputationalGraph::NodeOpsForward(std::string Nodeid)
{
    if(!CheckOps(Nodeid))return;
    GetCGOps(Nodeid)->Forward();
}

void ComputationalGraph::ForwardDfs(std::string DfsStartNodeid)
{
    if(ComputeFlag.find(DfsStartNodeid) != ComputeFlag.end() && ComputeFlag[DfsStartNodeid])return;
    ComputeFlag[DfsStartNodeid] = true;
    ComputationalNode* FoundNode = GetNode(DfsStartNodeid);
    for(int a =0;a<FoundNode->InputNodeidList.size();a++)
    {
        ForwardDfs(FoundNode->InputNodeidList[a]);
    }
    NodeOpsForward(DfsStartNodeid);
}

std::shared_ptr<BaseOps> ComputationalGraph::GetCGOps(std::string OpsNodeid)
{
    Log::Assert(CheckOps(OpsNodeid), std::string("This Node Has No Ops :")+OpsNodeid);
    return Opss[OpsNodeid];
}

void ComputationalGraph::ClearAllData()
{
    ClearDataPropertyExclude({});
}

bool ComputationalGraph::HasNode(std::string InputNode)
{
    return Nodes.find(std::string(InputNode))!=Nodes.end();
}

bool ComputationalGraph::HasDNode(std::string InputNode)
{
    return HasNode(GetDNodeid(InputNode));
}

void ComputationalGraph::SetAllNodeToInput()
{
    for(auto &NodePtr:Nodes)
    {
        static_cast<ComputationalNode*>(NodePtr.second)->Property.Set("Input", true);
    }
}

void ComputationalGraph::PrintGraphAdjacencyList(size_t Mode)
{
    std::cout<<"---------------Adjacency List----------------------"<<std::endl;
    for(auto &NodePtr:Nodes)
    {
        std::cout<<"Node Name :"<<NodePtr.first<<std::endl;
        if(Mode&1)
        {
            std::cout<<"Input Node :";
            for(auto& ItemNode:static_cast<ComputationalNode*>(NodePtr.second)->InputNodeidList)
            {
                std::cout<<ItemNode<<" ";
            }
            std::cout<<std::endl;
        }
        if(Mode&2)
        {
            std::cout<<"Output Node :";
            for(auto& ItemNode:static_cast<ComputationalNode*>(NodePtr.second)->OutputNodeidList)
            {
                std::cout<<ItemNode<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
}

bool ComputationalGraph::CheckInputNodeidCanBackward(std::string InputNodeid)
{
    if(CurrentBackwardFlag.find(InputNodeid) == CurrentBackwardFlag.end())return false;
    return CurrentBackwardFlag[InputNodeid] == CurrentBackwardFlagIndex;
}

void ComputationalGraph::BackwardMultiBuildGraph(size_t Times)
{
    CurrentBackwardFlag.clear();
    for(size_t a = 0;a<Times;a++)
    {
        CurrentBackwardFlagIndex = a;
        for(auto &NodePtr:Nodes)CurrentBackwardFlag[NodePtr.first] = CurrentBackwardFlagIndex;
        SetAllNodeToInput();
        BackwardGraphBuild();
    }
}

void ComputationalGraph::ClearDataPropertyExclude(std::vector<std::string>CheckPropertyList)
{
    for(auto &NodePtr:Nodes)
    {
        bool ClearFlag = false;
        for(auto &PropertyName:CheckPropertyList)
        {
            if(static_cast<ComputationalNode*>(NodePtr.second)->Property.Get<bool>(PropertyName) == true)
            {
                ClearFlag |= true;
                break;
            }
        }
        if(!ClearFlag)
        {
            static_cast<ComputationalNode*>(NodePtr.second)->ClearContent();
            ComputeFlag[NodePtr.first] = false;
        }
    }
}

void ComputationalGraph::ClearDataPropertyExclude()
{
    ClearDataPropertyExclude({"Weight", "Const"});
}