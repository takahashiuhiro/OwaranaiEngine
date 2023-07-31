#pragma once
#include "BaseNode.h"

class BaseEdge
{
public:
    virtual ~BaseEdge(){};
    std::string id;
    std::shared_ptr<BaseNode>S = nullptr;
    std::shared_ptr<BaseNode>E = nullptr;
};