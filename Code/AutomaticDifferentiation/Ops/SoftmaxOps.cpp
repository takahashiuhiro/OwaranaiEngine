#include "SoftmaxOps.h"
#include "../ComputationalGraph.h"

SoftmaxOps::SoftmaxOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    BaseOps::CommonInit(OpsTypeName, Params, ParentCG);
    ParamsDefinition();
}

void SoftmaxOps::Forward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("SoftmaxOps Must Have 1 Input Node"));
    Tensor* InputNode = this->CG->GetNode(NodeidList[0])->GetContent();
    auto SoftmaxDim = this->GetSelectDim(NodeidList[0]);
    this->CG->GetNode(this->Nodeid)->AssignContent(InputNode->Softmax(SoftmaxDim));
}

void SoftmaxOps::Backward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("SoftmaxOps Must Have 1 Input Node"));
    if(this->CG->GetNode(NodeidList[0])->Property.Get<bool>("RequireGrad"))
    {
        std::string ThisDNodeid = this->CG->GetDNodeid(this->Nodeid);
        std::string ThisEleMulThisDNodeid = this->CG->GetNodeidByOps(OpsType::EleMul, {this->Nodeid, ThisDNodeid});
        this->CG->RegisterVariableNode(ThisEleMulThisDNodeid);
        this->CG->RegisterOpsCompleted(ThisEleMulThisDNodeid, {this->Nodeid, ThisDNodeid}, OpsType::EleMul, Dict());
        this->CG->GetCGOps(ThisEleMulThisDNodeid)->AfterSettingShapeComputing();
        std::string ThisEleMulThisDNodeidSum = this->CG->GetNodeidByOps(OpsType::Sum, {ThisEleMulThisDNodeid});
        this->CG->RegisterVariableNode(ThisEleMulThisDNodeidSum);
        this->CG->RegisterOpsCompleted(ThisEleMulThisDNodeidSum, {ThisEleMulThisDNodeid}, OpsType::Sum, Dict());
        this->CG->GetCGOps(ThisEleMulThisDNodeidSum)->SetSelectDims({{ThisEleMulThisDNodeid,{this->GetSelectDim(NodeidList[0])}}});
        this->CG->GetCGOps(ThisEleMulThisDNodeidSum)->AfterSettingShapeComputing();
        std::string ThisEleMulThisDNodeidSumBroadCast = this->CG->GetNodeidByOps(OpsType::BroadCastTo, {ThisEleMulThisDNodeidSum});
        this->CG->RegisterVariableNode(ThisEleMulThisDNodeidSumBroadCast);
        this->CG->RegisterOpsCompleted(ThisEleMulThisDNodeidSumBroadCast, {ThisEleMulThisDNodeidSum}, OpsType::BroadCastTo, Dict());
        this->CG->GetCGOps(ThisEleMulThisDNodeidSumBroadCast)->SetBroadCastTo({{ThisEleMulThisDNodeidSum, this->CG->GetNode(this->Nodeid)->NodeContentShape}});
        this->CG->GetCGOps(ThisEleMulThisDNodeidSumBroadCast)->AfterSettingShapeComputing();
        std::string ThisMinusCast = this->CG->GetNodeidByOps(OpsType::Add, {ThisEleMulThisDNodeidSumBroadCast});
        this->CG->RegisterVariableNode(ThisMinusCast);
        this->CG->RegisterOpsCompleted(ThisMinusCast, {ThisDNodeid, ThisEleMulThisDNodeidSumBroadCast}, OpsType::Add, Dict());
        this->CG->GetCGOps(ThisMinusCast)->SetAddWeight({{ThisEleMulThisDNodeidSumBroadCast, -1.}});
        this->CG->GetCGOps(ThisMinusCast)->AfterSettingShapeComputing();
        std::string NewDNode = this->CG->GetDPartNodeid(NodeidList[0], Nodeid);
        this->CG->RegisterVariableNode(NewDNode);
        this->CG->RegisterOpsCompleted(NewDNode, {this->Nodeid, ThisMinusCast}, OpsType::EleMul, Dict());
        this->CG->GetCGOps(NewDNode)->AfterSettingShapeComputing();
        this->CG->RegisterOpsAddEdge(this->CG->GetDNodeid(NodeidList[0]), NewDNode);
        this->CG->GetCGOps(this->CG->GetDNodeid(NodeidList[0]))->SetAddWeight({{NewDNode, 1.}});
    }

}

void SoftmaxOps::AfterSettingShapeComputing()
{
    auto NodeidList = GetInputNodeList();
    this->CG->GetNode(this->Nodeid)->NodeContentShape = this->CG->GetNode(NodeidList[0])->NodeContentShape;
}