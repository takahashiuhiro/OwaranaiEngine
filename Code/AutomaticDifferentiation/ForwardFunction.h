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
    static std::string EleMul(ComputationalGraph*CG,std::string FirstNode, std::string SecondNode,float FirstAddWeight = 1., float SecondAddWeight = 1.);
    static std::string EleMul(std::shared_ptr<ComputationalGraph>CG,std::string FirstNode, std::string SecondNode,float FirstAddWeight = 1., float SecondAddWeight = 1.);
    /**矩阵乘法.*/
    static std::string MatMul(ComputationalGraph*CG, std::string FirstNode, std::string SecondNode, bool FirstTFlag = false, bool SecondTFlag = false, float FirstAddWeight = 1., float SecondAddWeight = 1.);
    static std::string MatMul(std::shared_ptr<ComputationalGraph>CG,std::string FirstNode, std::string SecondNode, bool FirstTFlag = false, bool SecondTFlag = false, float FirstAddWeight = 1., float SecondAddWeight = 1.);
    /**求和.*/
    static std::string Sum(ComputationalGraph*CG,std::string InputNode, std::vector<size_t>InputDims);
    static std::string Sum(std::shared_ptr<ComputationalGraph>CG,std::string InputNode, std::vector<size_t>InputDims);
    /**平均数.*/
    static std::string Mean(ComputationalGraph*CG,std::string InputNode, std::vector<size_t>InputDims);
    static std::string Mean(std::shared_ptr<ComputationalGraph>CG,std::string InputNode, std::vector<size_t>InputDims);
    /**方差.*/
    static std::string Var(ComputationalGraph*CG,std::string InputNode, std::vector<size_t>InputDims,bool Unbiased = true);
    static std::string Var(std::shared_ptr<ComputationalGraph>CG,std::string InputNode, std::vector<size_t>InputDims,bool Unbiased = true);
    /**ReLU.*/
    static std::string ReLU(ComputationalGraph*CG,std::string InputNode);
    static std::string ReLU(std::shared_ptr<ComputationalGraph>CG,std::string InputNode);
    /**Softmax.*/
    static std::string Softmax(ComputationalGraph*CG,std::string InputNode, size_t InputDim);
    static std::string Softmax(std::shared_ptr<ComputationalGraph>CG,std::string InputNode, size_t InputDim);
};

