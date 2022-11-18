#include "CGNode.h"

CGNode::CGNode(bool NeedGradient)
{
    this->NeedGradient = NeedGradient;
}

CGNode::CGNode(Tensor* NodeContent, bool NeedGradient)
{
    this->NodeContent = NodeContent;
    this->NeedGradient = NeedGradient;
}

CGNode::CGNode(std::string OpsType, bool NeedGradient)
{
    this->OpsType = OpsType;
    this->NeedGradient = NeedGradient;
    SetOps(OpsType);
}

CGNode::CGNode(std::vector<CGNode*>InputNode, std::string OpsType, bool NeedGradient)
{
    this->InputNode = InputNode;
    this->OpsType = OpsType;
    this->NeedGradient = NeedGradient;
    SetOps(OpsType);
}


void CGNode::Forward()
{
    if(InputNode.size() == 0 || NodeContent!=nullptr)return;
    for(int a=0;a<InputNode.size();a++)
    {
        InputNode[a]->Forward();
    }
    FunOps->Forward();
}

void CGNode::BackwardBuild(bool IsOutput)
{
    if(IsOutput)
    {
        BackwardBuildFlag = 1;
        DerivativeNode = new CGNode("Add", NeedGradient);
    }
    for(int a=0;a<InputNode.size();a++)
    {
        /**every node needs to buld its Gradient node without NeedGradient == 0*/
        if((InputNode[a]->NeedGradient == 0) || (InputNode[a]->BackwardBuildFlag == 1))continue;
        InputNode[a]->DerivativeNode = new CGNode("Add", NeedGradient);
        InputNode[a]->BackwardBuildFlag = 1;
        //std::cout<<InputNode[a]<<" aaaaaaaameshabi "<<InputNode[a]->BackwardBuildFlag<<" "<<this<<std::endl;
        //std::cout<<InputNode[a]->DerivativeNode<<" shenmeshabiiii\n";
    }
    //std::cout<<"kankan jici\n";
    FunOps->Backward();


    for(int a=0;a<InputNode.size();a++)
    {
        if(InputNode[a]->NeedGradient == 0 ||InputNode[a]->InputNode.size() == 0)continue;
        //std::cout<<InputNode[a]<<" capcapcapcapcpacapcapcap "<<InputNode[a]->BackwardBuildFlag<<"  "<<InputNode[a]->InputNode.size()<<" "<<std::endl;
        //InputNode[a]->BackwardBuildFlag = 1;
        InputNode[a]->BackwardBuild(0);
    }
}

void CGNode::Backward(Tensor* Loss)
{
    /**hajime no backward*/
    BackwardBuild(1);
    Tensor* DerivativeContent;
    Tensor* VectorTMP = new Tensor(std::vector<size_t>{1, (NodeContent->ShapeCount)/NodeContent->shape[0]}, NodeContent->Device, NodeContent->DeviceNum);
    VectorTMP->FillArray(1.);
    DerivativeContent = Loss->Matmul(VectorTMP);
    DerivativeContent->shape = NodeContent->shape;
    DerivativeNode->NodeContent = DerivativeContent;
}


void CGNode::SetOps(std::string OpsType)
{
    if(OpsType == "Add")
    {
        this->FunOps = new AddOps<CGNode, Tensor>(this);
    }
    else if(OpsType == "Matmul")
    {
        this->FunOps = new MatmulOps<CGNode, Tensor>(this);
    }
}

