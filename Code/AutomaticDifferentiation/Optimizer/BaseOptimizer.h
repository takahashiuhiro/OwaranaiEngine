#pragma once
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "../../CommonMathMoudle/Tensor.h"
#include "../../CommonDataStructure/Dict.h"
#include "../ComputationalGraph.h"

class ComputationalGraph;

class BaseOptimizer
{
public:

    std::shared_ptr<ComputationalGraph> CG = nullptr;

    /**从计算图中获取需要更新的计算节点.*/
    std::vector<std::string> GetWeightUpdateNodes();
    
};