#include "ComputationalNode.h"
#include "../CommonDataStructure/Log.h"

ComputationalNode::ComputationalNode()
{
    InitProperty();
}

ComputationalNode::ComputationalNode(std::string id)
{
    this->id = id;
    InitProperty();
}

ComputationalNode::~ComputationalNode()
{
    if(Content != nullptr)
    {
        delete Content;
    }
    OutputNodeidList.clear();
    InputNodeidList.clear();
}

void ComputationalNode::InitProperty()
{
    Property.Set("Input",false);//是否是原始输入图，允许梯度图为原始输入图
    Property.Set("RequireGrad", false);//是否需要梯度
    Property.Set("IsParamNode", false);//是否是参数矩阵
}

void ComputationalNode::ClearContent()
{
    if(Content!=nullptr)
    {
        delete Content;
    }
    Content = nullptr;
}

void ComputationalNode::AssignContent(Tensor* InputTensor)
{
   ClearContent();
   Content = InputTensor;
}

Tensor* ComputationalNode::GetContent()
{
    Log::Assert((Content!=nullptr), std::string("Content is nullptr, Node id:") + id);
    return Content;
}

void ComputationalNode::PrintData()
{
    GetContent()->PrintData();
}