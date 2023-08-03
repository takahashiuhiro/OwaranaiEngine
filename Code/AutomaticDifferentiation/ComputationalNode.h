#pragma once
#include <vector>
#include "../CommonDataStructure/BaseNode.h"
#include "../CommonMathMoudle/Tensor.h"
#include "../CommonDataStructure/Dict.h"



class ComputationalNode:public BaseNode
{
public:

    ComputationalNode();
    ComputationalNode(std::string);
    ~ComputationalNode();

    /**记录属性.*/
    Dict Property;
    void InitProperty();

    std::vector<std::string> OutputNodeidList;
    std::vector<std::string> InputNodeidList;

    Tensor* Content = nullptr;

    /**导数节点的id, 空的代表没有导数.*/
    std::string DNodeid = "";

    /**.*/
    void ClearContent();

};