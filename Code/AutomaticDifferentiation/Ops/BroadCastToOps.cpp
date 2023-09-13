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
    this->CG->GetNode(this->Nodeid)->AssignContent(FirstInputNode->BroadCastTo(BroadCastToPairShape));
}

void BroadCastToOps::Backward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("BroadCastToOps Must Have 1 Input Node"));
    std::string ThisDNodeid = this->CG->GetDNodeid(this->Nodeid);
    if(this->CG->GetNode(NodeidList[0])->Property.Get<bool>("RequireGrad"))
    {
        std::string NewDNode = this->CG->GetDPartNodeid(NodeidList[0], Nodeid);
        this->CG->RegisterVariableNode(NewDNode);
        this->CG->RegisterOps(NewDNode, std::vector<std::string>{ThisDNodeid}, OpsType::Sum, Dict());
        std::vector<size_t>SumDims;
        for(size_t a = 0;a < this->CG->GetNode(this->Nodeid)->NodeContentShape.size(); a++)
        {
            if(this->CG->GetNode(this->Nodeid)->NodeContentShape[a]!=this->CG->GetNode(NodeidList[0])->NodeContentShape[a])
            {
                SumDims.push_back(a);
            }
        }
        this->CG->GetCGOps(NewDNode)->SetSelectDims({{ThisDNodeid,SumDims}});
        this->CG->GetCGOps(NewDNode)->AfterSettingShapeComputing();
        this->CG->RegisterOpsAddEdge(this->CG->GetDNodeid(NodeidList[0]), NewDNode);
        this->CG->GetCGOps(this->CG->GetDNodeid(NodeidList[0]))->SetAddWeight({{NewDNode, 1.}});
    }
}

void BroadCastToOps::AfterSettingShapeComputing()
{
    auto NodeidList = GetInputNodeList();
    this->CG->GetNode(this->Nodeid)->NodeContentShape = this->GetBroadCastTo(NodeidList[0]);
}