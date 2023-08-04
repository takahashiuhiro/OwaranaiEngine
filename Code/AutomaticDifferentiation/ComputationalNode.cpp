#include "ComputationalNode.h"

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

void ComputationalNode::AssertContentNullptr()
{
    assert((Content!=nullptr) && "Content is nullptr!!!");
}

Tensor* ComputationalNode::GetContent()
{
    AssertContentNullptr();
    return Content;
}

void ComputationalNode::PrintData()
{
    GetContent()->PrintData();
}