#pragma once
#include <iostream>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "../../CommonMathMoudle/Tensor.h"
#include "../../CommonDataStructure/Dict.h"
#include "../ComputationalGraph.h"
#include "../Ops/OpsType.h"

class BaseLayer
{
public:
    /**树上的链.*/
    std::string PreName = "";
    /**子节点们.*/
    std::map<std::string, std::shared_ptr<BaseLayer>>SubLayers;//todo::别忘了这里要释放内存
    /**root的计算图，只允许root是非nullptr.*/
    std::shared_ptr<ComputationalGraph> CG = nullptr;
    /**父节点，root为nullptr.*/
    BaseLayer* ParentLayer = nullptr;

    /**判断该层是否为root节点.*/
    bool IsRootNode();
    /**公共init，例如在构造函数的时候声明计算图等.调用这个函数前要先把该分给子层的分了，防止重复声明计算图，寄了*/
    void CommonInit(BaseLayer* InputParentLayer);

    /**对标torch的init模块，是用来声明各种参数矩阵和常数矩阵的.,参数输入不在这*/
    virtual void Init(){};
    /**进行前向构建的.*/
    virtual void Forward(){};
    /**感觉应该还得有一个启动函数.*/
    virtual void Run(){};
};