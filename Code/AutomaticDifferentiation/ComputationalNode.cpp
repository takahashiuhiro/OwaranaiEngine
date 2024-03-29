#include "ComputationalNode.h"
#include "../CommonDataStructure/Log.h"

ComputationalNode::ComputationalNode(std::string id)
{
    this->id = id;
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

void ComputationalNode::PrintShape()
{
    GetContent()->PrintShape();
}

void ComputationalNode::PrintNodeContentShape()
{
    std::cout<<"NodeContentShape:{";
    for(size_t a=0;a<NodeContentShape.size();a++)
    {
        std::cout<<" "<<NodeContentShape[a];
    }
    std::cout<<" }"<<std::endl;
}