#pragma once
#include <vector>
#include "../CommonDataStructure/BaseNode.h"
#include "../CommonMathMoudle/Tensor.h"
#include "../CommonDataStructure/Dict.h"

class ComputationalNode:public BaseNode
{
public:

    ComputationalNode(){};
    ComputationalNode(std::string);
    ~ComputationalNode();

    /**记录属性.*/
    Dict Property;

    std::vector<std::string> OutputNodeidList;
    std::vector<std::string> InputNodeidList;
    std::vector<size_t>NodeContentShape;

    Tensor* Content = nullptr;

    /**导数节点的id, 空的代表没有导数.*/
    std::string DNodeid = "";
    /**清理节点的Tensor内容.*/
    void ClearContent();
    /**赋值函数，不能直接调用content相等，会泄内存.*/
    void AssignContent(Tensor* InputTensor);
    /**返回content，包一个nullptr的检查.*/
    Tensor* GetContent();
    /**打印数据.*/
    void PrintData();
    /**打印形态.*/
    void PrintShape();
    /**打印节点应该出现的形态.*/
    void PrintNodeContentShape();
};