#include "ViewOps.h"
#include "../ComputationalGraph.h"

ViewOps::ViewOps(size_t OpsTypeName, Dict Params, ComputationalGraph* ParentCG)
{
    ViewOps::CommonInit(OpsTypeName, Params, ParentCG);
    ParamsDefinition();
}

void ViewOps::Forward()
{
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("ViewOps Must Have 1 Input Node"));
    Tensor* InputNode = this->CG->GetNode(NodeidList[0])->GetContent();
    auto ViewShape = this->GetBroadCastTo(NodeidList[0]);
    int MinusShape = this->GetSelectDimSingle();
    this->CG->GetNode(this->Nodeid)->AssignContent(InputNode->View(ViewShape, MinusShape));
}

void ViewOps::Backward()
{   
    auto NodeidList = GetInputNodeList();
    Log::Assert(NodeidList.size() == 1, std::string("ViewOps Must Have 1 Input Node"));
    std::string ThisDNodeid = this->CG->GetDNodeid(this->Nodeid);
    if(this->CG->GetNode(NodeidList[0])->Property.Get<bool>("RequireGrad"))
    {
        std::string NewDNode = OEAutoDiff::View(this->CG, ThisDNodeid, this->CG->GetNode(NodeidList[0])->NodeContentShape, -1);
        this->CG->RegisterOpsAddEdge(this->CG->GetDNodeid(NodeidList[0]), NewDNode);
        this->CG->GetCGOps(this->CG->GetDNodeid(NodeidList[0]))->SetAddWeight({{NewDNode, 1.}});
    }
}

void ViewOps::AfterSettingShapeComputing()
{
    auto NodeidList = GetInputNodeList();
    auto ViewShape = this->GetBroadCastTo(NodeidList[0]);
    int MinusShape = this->GetSelectDimSingle();
    int SpcIdx = 1;
    for(size_t a =0;a<this->CG->GetNode(NodeidList[0])->NodeContentShape.size();a++)SpcIdx*=this->CG->GetNode(NodeidList[0])->NodeContentShape[a];
    this->CG->GetNode(this->Nodeid)->NodeContentShape = ViewShape;
    if(MinusShape >= 0)this->CG->GetNode(this->Nodeid)->NodeContentShape[MinusShape] = SpcIdx/this->CG->GetNode(NodeidList[0])->NodeContentShape[MinusShape];
}