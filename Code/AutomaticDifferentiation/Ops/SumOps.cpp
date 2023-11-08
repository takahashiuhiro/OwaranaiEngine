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
        std::string NewDNode = OEAutoDiff::BroadCastTo(this->CG, ThisDNodeid, this->CG->GetNode(NodeidList[0])->NodeContentShape);
        this->CG->RegisterOpsAddEdge(this->CG->GetDNodeid(NodeidList[0]), NewDNode);
        this->CG->GetCGOps(this->CG->GetDNodeid(NodeidList[0]))->SetAddWeight({{NewDNode, 1.}});
    }
}

void SumOps::AfterSettingShapeComputing()
{
    auto NodeidList = GetInputNodeList();
    auto SumDims = this->GetSelectDims(NodeidList[0]);
    this->CG->GetNode(this->Nodeid)->NodeContentShape = this->CG->GetNode(NodeidList[0])->NodeContentShape;
    for(size_t a=0;a<SumDims.size();a++)
    {
        this->CG->GetNode(this->Nodeid)->NodeContentShape[SumDims[a]] = 1;
    }
}