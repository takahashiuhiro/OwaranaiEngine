#include "PowOps.h"
#include "../ComputationalGraph.h"

PowOps::PowOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    BaseOps::CommonInit(OpsTypeName, Params, ParentCG);
    ParamsDefinition();
}

void PowOps::Forward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("PowOps Must Have 1 Input Node"));
    Tensor* InputNode = this->CG->GetNode(NodeidList[0])->GetContent();
    this->CG->GetNode(this->Nodeid)->AssignContent(InputNode->Pow(GetEleExponent()));
}

void PowOps::Backward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("PowOps Must Have 1 Input Node"));
    if(this->CG->GetNode(NodeidList[0])->Property.Get<bool>("RequireGrad"))
    {
        std::string ThisDNodeid = this->CG->GetDNodeid(Nodeid);
        std::string PowNode = OEAutoDiff::Pow(this->CG, NodeidList[0], GetEleExponent()-1);
        std::string NewDNode = this->CG->GetDPartNodeid(NodeidList[0], Nodeid);
        this->CG->RegisterVariableNode(NewDNode);
        this->CG->RegisterOpsCompleted(NewDNode, {ThisDNodeid, PowNode}, OpsType::EleMul, Dict());
        this->CG->GetCGOps(NewDNode)->AfterSettingShapeComputing();
        this->CG->RegisterOpsAddEdge(this->CG->GetDNodeid(NodeidList[0]), NewDNode);
        this->CG->GetCGOps(this->CG->GetDNodeid(NodeidList[0]))->SetAddWeight({{NewDNode, GetEleExponent()}});
    }
}

void PowOps::AfterSettingShapeComputing()
{
    auto NodeidList = GetInputNodeList();
    this->CG->GetNode(this->Nodeid)->NodeContentShape = this->CG->GetNode(NodeidList[0])->NodeContentShape;
}