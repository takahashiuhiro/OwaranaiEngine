#pragma once
#include <iostream>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "../../CommonDataStructure/CommonFuncHelpers.h"
#include "../../CommonMathMoudle/Tensor.h"
#include "../../CommonDataStructure/Dict.h"
#include "../ComputationalGraph.h"
#include "../Ops/OpsType.h"

class BaseLoss
{
public:
    std::shared_ptr<ComputationalGraph>CG = nullptr;
    std::vector<std::string> LossNodes;

    void CommonInit(std::shared_ptr<ComputationalGraph>InputCG);
    void SetLossData(std::map<std::string, Tensor*>LossMap);

    /**封装误差反传流程.*/
    virtual void Backward();

    /**在计算图上建立loss节点,返回loss节点.*/
    virtual void Build(std::vector<std::string>InputCGNodeList, std::vector<std::string>LabelNodeList) = 0;
};