#include "BroadCastToOps.h"
#include "../ComputationalGraph.h"

BroadCastToOps::BroadCastToOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    BroadCastToOps::CommonInit(OpsTypeName, Params, ParentCG);
    ParamsDefinition();
}

void BroadCastToOps::Forward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("BroadCastToOps Must Have 1 Input Node"));
    Tensor* FirstInputNode = this->CG->GetNode(NodeidList[0])->GetContent();
    auto BroadCastToPairShape = this->GetBroadCastTo(NodeidList[0]);
    this->CG->GetNode(this->Nodeid)->AssignContent(FirstInputNode->BroadCastTo(BroadCastToPairShape[1]));
}

void BroadCastToOps::Backward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("BroadCastToOps Must Have 1 Input Node"));
    std::string ThisDNodeid = this->CG->GetDNodeid(this->Nodeid);
    if(this->CG->GetNode(NodeidList[0])->Property.Get<bool>("RequireGrad"))
    {
        
    }
}