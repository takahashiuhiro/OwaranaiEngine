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
CGNode::CGNode(std::string OpsType, bool NeedGradient, Hyperparameter OpsParams)
{
    this->OpsType = OpsType;
    this->NeedGradient = NeedGradient;
    SetOps(OpsType, OpsParams);
}

CGNode::CGNode(std::vector<CGNode*>InputNode, std::string OpsType, bool NeedGradient)
{
    this->InputNode = InputNode;
    this->OpsType = OpsType;
    this->NeedGradient = NeedGradient;
    SetOps(OpsType);
}

CGNode::CGNode(std::vector<CGNode*>InputNode, std::string OpsType, bool NeedGradient, Hyperparameter OpsParams)
{
    this->InputNode = InputNode;
    this->OpsType = OpsType;
    this->NeedGradient = NeedGradient;
    SetOps(OpsType, OpsParams);
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
    if(BackwardBuildFlag)return;
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

void CGNode::ClearDataContent(std::vector<std::string>NodeTypeList, bool IsInclude)
{
    if(IsInclude)
    {
        //????????????nodetypelist???????????????????????????
        for(int a =0 ;a<NodeTypeList.size();a++)
        {
            if(!NodeType.count(NodeTypeList[a]))continue;
            delete NodeContent;
            NodeContent = nullptr;
            break;
        }
    }
    else
    {
        //????????????nodetypelist??????????????????????????????????????????
        for(int a =0 ;a<NodeTypeList.size();a++)
        {
            if(NodeType.count(NodeTypeList[a]))return;
        }
        delete NodeContent;
        NodeContent = nullptr;
    }
}

void CGNode::ClearDataDFS(std::vector<std::string>NodeTypeList, bool IsInclude, std::map<CGNode*, bool>*FlagMap)
{
    //NodeTypeList?????????????????????????????????????????????????????????????????????????????????????????????????????????
    //????????????????????????????????????
    if(NeedGradient == 0)return;
    //???????????????dfs
    if(NodeContent == nullptr)return;
    if(FlagMap->count(this))return;
    (*FlagMap)[this] = 1;
    //std::cout<<"?????????????????????????????? ???!"<<NodeContent<<std::endl;
    ClearDataContent(NodeTypeList, IsInclude);
    //std::cout<<"?????????????????????????????? ???!"<<NodeContent<<std::endl;
    for(int a=0;a<InputNode.size();a++)
    {
        InputNode[a]->ClearDataDFS(NodeTypeList, IsInclude, FlagMap);
    }
}

void CGNode::ClearGradient(std::vector<CGNode*>InputNodeList)
{
    std::map<CGNode*, bool>FlagMap;
    for(int a=0;a<InputNodeList.size();a++)
    {
        InputNodeList[a]->DerivativeNode->ClearDataDFS(std::vector<std::string>{"Gradient"}, 1, &FlagMap);
    }
}

void CGNode::SetOps(std::string OpsType)
{
    //?????????????????????
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
    else if(OpsType == "Addarray")
    {
        this->FunOps = new AddarrayOps<CGNode, Tensor>(this);
    }
}

void CGNode::SetOps(std::string OpsType, Hyperparameter OpsParams)
{
    if (OpsType == "Flatten")
    {
        this->FunOps = new FlattenOps<CGNode, Tensor>(this, OpsParams);
    }
}

