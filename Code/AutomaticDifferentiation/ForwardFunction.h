#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "Ops/OpsType.h"

class ComputationalGraph;

class OEAutoDiff
{
public:
    /**平均数.*/
    static std::vector<std::string> Mean(ComputationalGraph*CG,std::vector<std::string>InputNodes, std::vector<size_t>InputDims);
    static std::vector<std::string> Mean(std::shared_ptr<ComputationalGraph>CG,std::vector<std::string>InputNodes, std::vector<size_t>InputDims);
};

