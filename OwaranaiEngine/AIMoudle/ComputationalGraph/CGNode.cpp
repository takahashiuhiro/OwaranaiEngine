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

void CGNode::Backward(std::string BackType, float Loss)
{
    //todo::这里稍微有点问题，待思考....Loss不应该是float的标量，他其实应该是向量，长为batch，这里forward部分也应该检查一下，因为按照矩阵广播有的参数矩阵也会被改shape
    //todo::想达成一个兼容多batch的反向传播，同时参数矩阵的shape不会改变
    //todo::我们可以限定loss就是向量，不是张量，并且梯度的shape必须等同于本体node的shape
    if(BackType == "Output")
    {
        /**hajime no backward*/
        Tensor* DerivativeContent = new Tensor(NodeContent->shape, NodeContent->Device, NodeContent->DeviceNum);
        DerivativeContent->FillArray(Loss);
        DerivativeNode = new CGNode(DerivativeContent, 1);
    }
    for(int a=0;a< InputNode.size();a++)
    {
        /**build DerivativeNode for every input node which has no DerivativeNode*/
        if(InputNode[a]->DerivativeNode != nullptr)
        {
            Tensor* DerivativeContent = new Tensor(InputNode[a]->NodeContent->shape, InputNode[a]->NodeContent->Device, InputNode[a]->NodeContent->DeviceNum);
            DerivativeContent->FillArray(0);
            InputNode[a]->DerivativeNode = new CGNode(DerivativeContent, 1);
        }
    }
    FunOps->Backward();
}


void CGNode::SetOps(std::string OpsType)
{
    if(OpsType == "Add")
    {
        this->FunOps = new AddOps<CGNode>(this);
    }
}

