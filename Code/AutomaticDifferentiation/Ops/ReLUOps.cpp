#include "ReLUOps.h"
#include "../ComputationalGraph.h"

ReLUOps::ReLUOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    BaseOps::CommonInit(OpsTypeName, Params, ParentCG);
    ParamsDefinition();
}

void ReLUOps::Forward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("ReluOps Must Have 1 Input Node"));
    Tensor* InputNode = this->CG->GetNode(NodeidList[0])->GetContent();
    this->CG->GetNode(this->Nodeid)->AssignContent(InputNode->ReLU());
}

void ReLUOps::Backward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("ReluOps Must Have 1 Input Node"));
    if(this->CG->GetNode(NodeidList[0])->Property.Get<bool>("RequireGrad"))
    {
        std::string SignNodeid = this->CG->GetNodeidByOps(OpsType::GenerateSign, {NodeidList[0]});
        this->CG->RegisterVariableNode(SignNodeid);
        this->CG->RegisterOpsCompleted(SignNodeid, {NodeidList[0]}, OpsType::GenerateSign, Dict());
        this->CG->GetCGOps(SignNodeid)->AfterSettingShapeComputing();

        std::string ThisDNodeid = this->CG->GetDNodeid(this->Nodeid);

        std::string ThisEleMulThisDNodeid = this->CG->GetDPartNodeid(NodeidList[0], this->Nodeid);
        this->CG->RegisterVariableNode(ThisEleMulThisDNodeid);
        this->CG->RegisterOpsCompleted(ThisEleMulThisDNodeid, {SignNodeid, ThisDNodeid}, OpsType::EleMul, Dict());
        this->CG->GetCGOps(ThisEleMulThisDNodeid)->AfterSettingShapeComputing();

        this->CG->RegisterOpsAddEdge(this->CG->GetDNodeid(NodeidList[0]), ThisEleMulThisDNodeid);
        this->CG->GetCGOps(this->CG->GetDNodeid(NodeidList[0]))->SetAddWeight({{ThisEleMulThisDNodeid, 1.}});
    }
}

void ReLUOps::AfterSettingShapeComputing()
{
    auto NodeidList = GetInputNodeList();
    this->CG->GetNode(this->Nodeid)->NodeContentShape = this->CG->GetNode(NodeidList[0])->NodeContentShape;
}