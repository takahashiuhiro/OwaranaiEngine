#include "CGNode.h"

CGNode::CGNode(Tensor* NodeContent, bool NeedGradient)
{
    this->NodeContent = NodeContent;
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

void CGNode::Backward(std::string BackType, Tensor* Loss)
{
    if(BackType == "Output")
    {
        /**hajime no backward*/
        Tensor* DerivativeContent;
        Tensor* VectorTMP = new Tensor(std::vector<size_t>{1, (NodeContent->ShapeCount)/NodeContent->shape[0]}, NodeContent->Device, NodeContent->DeviceNum);
        VectorTMP->FillArray(1.);
        DerivativeContent = Loss->Matmul(VectorTMP);
        DerivativeContent->shape = NodeContent->shape;
        DerivativeNode = new CGNode(DerivativeContent, 1);
    }
    FunOps->Backward();
}


void CGNode::SetOps(std::string OpsType)
{
    if(OpsType == "Add")
    {
        this->FunOps = new AddOps<CGNode, Tensor>(this);
    }
}

