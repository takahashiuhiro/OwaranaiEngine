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
        DerivativeNode = new CGNode("Add", NeedGradient);
        DerivativeNode->NodeType["Gradient"] = 1;
    }
    for(int a=0;a<InputNode.size();a++)
    {
        /**every node needs to buld its Gradient node without NeedGradient == 0*/
        if((InputNode[a]->NeedGradient == 0) || (InputNode[a]->DerivativeNode != nullptr))continue;
        InputNode[a]->DerivativeNode = new CGNode("Add", NeedGradient);
        InputNode[a]->DerivativeNode->NodeType["Gradient"] = 1;
    }
    if(BackwardBuildFlag)return;
    FunOps->Backward();
    BackwardBuildFlag = 1;

    for(int a=0;a<InputNode.size();a++)
    {
        if(InputNode[a]->NeedGradient == 0 ||InputNode[a]->InputNode.size() == 0)continue;
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

void CGNode::ClearDataContent(std::vector<std::string>NodeTypeList)
{
    //删除包含nodetypelist中标签的节点的内容
    for(int a =0 ;a<NodeTypeList.size();a++)
    {
        if(!NodeType.count(NodeTypeList[a]))continue;
        delete NodeContent;
        NodeContent = nullptr;
        break;
    }
}

void CGNode::ClearDataDFS(std::vector<std::string>NodeTypeList)
{
    //NodeTypeList最大的作用应该是区别输入张量，前向张量，参数张量，常数张量，导数张量的
    //检测该节点是否是常数张量
    if(NeedGradient == 0)return;
    //中断多余的dfs
    if(NodeContent == nullptr)return;
    //检测该节点是否带有保护标签
    for(int a =0 ;a<NodeTypeList.size();a++)
    {
        if(NodeType.count(NodeTypeList[a]))
        {
            return;
        }
    }
    //std::cout<<"你别告诉我你是空的啊 开!"<<NodeContent<<std::endl;
    delete NodeContent;
    NodeContent = nullptr;
    //std::cout<<"你别告诉我你是空的啊 关!"<<NodeContent<<std::endl;
    for(int a=0;a<InputNode.size();a++)
    {
        InputNode[a]->ClearDataDFS(NodeTypeList);
    }
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
    else if(OpsType == "MatmulFirstT")
    {
        this->FunOps = new MatmulFirstTOps<CGNode, Tensor>(this);
    }
    else if(OpsType == "MatmulSecondT")
    {
        this->FunOps = new MatmulSecondTOps<CGNode, Tensor>(this);
    }
    else if(OpsType == "Elemul")
    {
        this->FunOps = new ElemulOps<CGNode, Tensor>(this);
    }
}

