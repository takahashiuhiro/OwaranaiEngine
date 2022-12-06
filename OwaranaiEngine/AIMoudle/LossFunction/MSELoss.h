#pragma once
#include "BaseLoss.h"

struct MSELoss:BaseLoss
{
public:
    virtual void LossBuild()
    {
        //todo::等下补好loss，还得写个减法算子..
        //CGNode *Intermediate_1 = new CGNode(std::vector<CGNode*>{OutputNode[0],LabelNode[0]},"Matmul", 1);
    }
};
