#include "EleExpOps.h"
#include "../ComputationalGraph.h"

EleExpOps::EleExpOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    BaseOps::CommonInit(OpsTypeName, Params, ParentCG);
    ParamsDefinition();
}

void EleExpOps::Forward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("EleExpOps Must Have 1 Input Node"));
    Tensor* InputNode = this->CG->GetNode(NodeidList[0])->GetContent();
    this->CG->GetNode(this->Nodeid)->AssignContent(InputNode->EleExp(GetEleBaseNum()));
}

void EleExpOps::Backward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("EleExpOps Must Have 1 Input Node"));
    if(this->CG->GetNode(NodeidList[0])->Property.Get<bool>("RequireGrad"))
    {
        float DotK = std::log(GetEleBaseNum());
        std::string ThisDNodeid = this->CG->GetDNodeid(Nodeid);
        std::string EleMulNode = OEAutoDiff::EleMul(this->CG, Nodeid, ThisDNodeid, 1, DotK);
        this->CG->RegisterOpsAddEdge(this->CG->GetDNodeid(NodeidList[0]), EleMulNode);
        this->CG->GetCGOps(this->CG->GetDNodeid(NodeidList[0]))->SetAddWeight({{EleMulNode, 1.}});
    }
}

void EleExpOps::AfterSettingShapeComputing()
{
    auto NodeidList = GetInputNodeList();
    this->CG->GetNode(this->Nodeid)->NodeContentShape = this->CG->GetNode(NodeidList[0])->NodeContentShape;
}