#include "EleMulOps.h"
#include "../ComputationalGraph.h"

EleMulOps::EleMulOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    BaseOps::CommonInit(OpsTypeName, Params, ParentCG);
    ParamsDefinition();
}

void EleMulOps::Forward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 2, std::string("EleMulOps Must Have 2 Input Node"));
    Tensor* FirstInputNode = this->CG->GetNode(NodeidList[0])->GetContent();
    Tensor* SecondInputNode = this->CG->GetNode(NodeidList[1])->GetContent();
    std::shared_ptr<Tensor> NodeRes = std::shared_ptr<Tensor>(FirstInputNode->EleMul(SecondInputNode));
    float AddWeightDot = GetAddWeight(NodeidList[0])*GetAddWeight(NodeidList[1]);
    this->CG->GetNode(this->Nodeid)->AssignContent(NodeRes->MulScalar(AddWeightDot));
}

void EleMulOps::BuildSingleGrad(std::string FirstNodeid, std::string SecondNodeid)
{
    std::string NewDNode = this->CG->GetDPartNodeid(FirstNodeid, Nodeid);
    this->CG->RegisterVariableNode(NewDNode);
    this->CG->RegisterOps(NewDNode, std::vector<std::string>{this->CG->GetDNodeid(Nodeid), SecondNodeid}, OpsType::EleMul, Dict());
    auto NodeidList = GetInputNodeList();
    float AddWeightDot = GetAddWeight(NodeidList[0])*GetAddWeight(NodeidList[1]);
    this->CG->GetCGOps(NewDNode)->SetAddWeight({{this->CG->GetDNodeid(Nodeid), 1.},{SecondNodeid, AddWeightDot}});
    this->CG->RegisterOpsAddEdge(this->CG->GetDNodeid(FirstNodeid), NewDNode);
    this->CG->GetCGOps(this->CG->GetDNodeid(FirstNodeid))->SetAddWeight({{NewDNode, 1.}});
}

void EleMulOps::Backward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 2, std::string("EleMulOps Must Have 2 Input Node"));
    std::string ThisDNodeid = this->CG->GetDNodeid(this->Nodeid);
    if(this->CG->GetNode(NodeidList[0])->Property.Get<bool>("RequireGrad"))
    {
        BuildSingleGrad(NodeidList[0], NodeidList[1]);
    }
    if(this->CG->GetNode(NodeidList[1])->Property.Get<bool>("RequireGrad"))
    {
        BuildSingleGrad(NodeidList[1], NodeidList[0]);
    }
}

void EleMulOps::AfterSettingShapeComputing()
{
    auto NodeidList = GetInputNodeList();
    this->CG->GetNode(this->Nodeid)->NodeContentShape = this->CG->GetNode(NodeidList[0])->NodeContentShape;
}