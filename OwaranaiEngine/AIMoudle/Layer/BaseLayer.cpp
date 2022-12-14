#include "BaseLayer.h"

void BaseLayer::LayerInit(std::vector<CGNode*>InputCGNode)
{
    this->InputCGNode = InputCGNode;
}

void BaseLayer::Freeze()
{
    for(int a =0;a<ParamsCGNode.size();a++)
    {
        ParamsCGNode[a]->NodeType["Freeze"] = 1;
    }
}

CGNode* BaseLayer::Forward()
{
    ForwardNode->Forward();
    return ForwardNode;
}

void BaseLayer::Backward(Tensor* Loss)
{
    LossFunctionPointer->LossNode->Backward(Loss);
    for(int a=0;a<ParamsCGNode.size();a++)
    {
        ParamsCGNode[a]->DerivativeNode->Forward();
    }
}

void BaseLayer::SetLossFunction(std::string LossType)
{
    if(LossType == "MSE")
    {
        LossFunctionPointer = new MSELoss();
    }
}
