#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include "Ops/OpsType.h"

class ComputationalGraph;

class OEAutoDiff
{
public:
    /**加法.*/
    static std::string Add(ComputationalGraph*CG,std::map<std::string, float> InputWeight);
    static std::string Add(std::shared_ptr<ComputationalGraph>CG,std::map<std::string, float> InputWeight);
    /**幂次.*/
    static std::string Pow(ComputationalGraph*CG,std::string InputNode,float Exponent);
    static std::string Pow(std::shared_ptr<ComputationalGraph>CG,std::string InputNode,float Exponent);
    /**广播.*/
    static std::string BroadCastTo(ComputationalGraph*CG,std::string InputNode,std::vector<size_t>InputDims);
    static std::string BroadCastTo(std::shared_ptr<ComputationalGraph>CG,std::string InputNode,std::vector<size_t>InputDims);
    /**元素乘法.*/
    static std::string EleMul(ComputationalGraph*CG,std::map<std::string, float> InputWeight);
    static std::string EleMul(std::shared_ptr<ComputationalGraph>CG,std::map<std::string, float> InputWeight);
    /**求和.*/
    static std::string Sum(ComputationalGraph*CG,std::string InputNode, std::vector<size_t>InputDims);
    static std::string Sum(std::shared_ptr<ComputationalGraph>CG,std::string InputNode, std::vector<size_t>InputDims);
    /**平均数.*/
    static std::string Mean(ComputationalGraph*CG,std::string InputNode, std::vector<size_t>InputDims);
    static std::string Mean(std::shared_ptr<ComputationalGraph>CG,std::string InputNode, std::vector<size_t>InputDims);
    /**方差.*/
    static std::string Var(ComputationalGraph*CG,std::string InputNode, std::vector<size_t>InputDims,bool Unbiased = true);
    static std::string Var(std::shared_ptr<ComputationalGraph>CG,std::string InputNode, std::vector<size_t>InputDims,bool Unbiased = true);

};

