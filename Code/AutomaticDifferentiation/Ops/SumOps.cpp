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
    std::string ThisDNodeid = this->CG->GetDNodeid(this->Nodeid);
    //todo
}