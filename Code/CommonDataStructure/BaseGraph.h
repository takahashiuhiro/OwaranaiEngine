#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <map>
#include <vector>
#include "BaseNode.h"
#include "BaseEdge.h"

class BaseGraph
{
public:
    BaseGraph(){};
    virtual ~BaseGraph(){};
    std::string id;
    std::map<std::string, BaseNode*>Nodes;
    std::map<std::string, std::map<std::string, BaseEdge>>Edges;
    /***/
    virtual void AddNode(BaseNode* NewNode) = 0;
    virtual void AddEdge(BaseEdge* NewEdge) = 0;

};


