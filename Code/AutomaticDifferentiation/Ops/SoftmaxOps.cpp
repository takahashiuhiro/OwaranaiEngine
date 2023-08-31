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
    
}