#include "CGNode.h"

CGNode::CGNode(Tensor* NodeContent, bool NeedGradient)
{
    this->NodeContent = NodeContent;
    this->Gradient = new Tensor(NodeContent->shape, NodeContent->Device, NodeContent->DeviceNum);
    this->Gradient->FillArray(0.);
    this->OpsType = "Input";
    this->NeedGradient = NeedGradient;
}

CGNode::CGNode(std::vector<CGNode*>InputNode, std::string OpsType, bool NeedGradient)
{
    this->InputNode = InputNode;
    this->OpsType = OpsType;
    this->NeedGradient = NeedGradient;
    SetOps(OpsType);
    FunOps->Forward();
    this->Gradient = new Tensor(NodeContent->shape, NodeContent->Device, NodeContent->DeviceNum);
    this->Gradient->FillArray(0.);
}


void CGNode::Forward()
{
    if(OpsType == "Input")return;
    for(int a=0;a<InputNode.size();a++)
    {
        InputNode[a]->Forward();
    }
    FunOps->Forward();
}

void CGNode::Backward()
{
    
}


void CGNode::SetOps(std::string OpsType)
{
    if(OpsType == "Add")
    {
        this->FunOps = new AddOps<CGNode>(this);
    }
}

