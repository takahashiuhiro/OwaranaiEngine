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
    /**清理节点的Tensor内容.*/
    void ClearContent();
    /**赋值函数，不能直接调用content相等，会泄内存.*/
    void AssignContent(Tensor* InputTensor);
    /**试图获取一个nullptr的时候中的断言，一定不对.*/
    void AssertContentNullptr();
    /**返回content，包一个nullptr的检查.*/
    Tensor* GetContent();
    /**打印数据.*/
    void PrintData();

};