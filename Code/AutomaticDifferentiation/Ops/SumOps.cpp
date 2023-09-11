#include "SumOps.h"
#include "../ComputationalGraph.h"

SumOps::SumOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    SumOps::CommonInit(OpsTypeName, Params, ParentCG);
    ParamsDefinition();
}

void SumOps::Forward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("SumOps Must Have 1 Input Node"));
    Tensor* InputNode = this->CG->GetNode(NodeidList[0])->GetContent();
    auto SumDims = this->GetSelectDims(NodeidList[0]);
    this->CG->GetNode(this->Nodeid)->AssignContent(InputNode->Sum(SumDims));
}

void SumOps::Backward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("SumOps Must Have 1 Input Node"));
    std::string ThisDNodeid = this->CG->GetDNodeid(this->Nodeid);
    if(this->CG->GetNode(NodeidList[0])->Property.Get<bool>("RequireGrad"))
    {
        std::string NewDNode = this->CG->GetDPartNodeid(NodeidList[0], Nodeid);
        this->CG->RegisterVariableNode(NewDNode);
        this->CG->RegisterOps(NewDNode, std::vector<std::string>{ThisDNodeid}, OpsType::BroadCastTo, Dict());
        auto SumShape = this->GetBroadCastTo(NodeidList[0]);
        this->CG->GetCGOps(NewDNode)->SetBroadCastTo({{ThisDNodeid, {SumShape[1], SumShape[0]}}});
        this->CG->RegisterOpsAddEdge(this->CG->GetDNodeid(NodeidList[0]), NewDNode);
        this->CG->GetCGOps(this->CG->GetDNodeid(NodeidList[0]))->SetAddWeight({{NewDNode, 1.}});
    }
}