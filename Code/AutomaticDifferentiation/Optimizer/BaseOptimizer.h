#pragma once
#include <vector>
#include <string>
#include <map>
#include "../../CommonMathMoudle/Tensor.h"
#include "../../CommonDataStructure/Dict.h"

class ComputationalGraph;

class BaseOptimizer
{
public:

    ComputationalGraph* CG;

    std::vector<std::string> GetWeightUpdateNode();
};