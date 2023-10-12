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
    std::vector<std::string> GetWeightUpdateNodes();
};